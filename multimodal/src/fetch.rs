// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resolving media sources to raw bytes.
//!
//! Two layers live here, with different trust levels:
//!
//! * [`MediaFetcher`]: a **protected** fetcher for untrusted request URLs
//!   (`http`, `https`, `data:` only). It refuses private and internal
//!   destinations, revalidates every redirect hop, filters DNS answers so a
//!   rebinding hostname cannot reach a blocked address, and caps the number of
//!   bytes read. The policy is a [`FetchPolicy`] fixed when the fetcher is built, and the fetcher owns the
//!   HTTP client that enforces it, so the two cannot drift apart.
//! * [`fetch_bytes`] and friends: a planned trusted-source helper (parity
//!   anchor: `transformers.image_utils.load_image`, which also reads local
//!   files). These are **still stubs**: they always return
//!   [`MmError::Unsupported`](crate::MmError::Unsupported). They are not a
//!   security boundary; do not call them on untrusted input.
//!
//! Nothing here reads environment variables, the HTTP client included: the
//! on-prem opt-ins and the proxy are plain [`FetchPolicy`] fields
//! ([`FetchPolicy::with_internal_access`], [`FetchPolicy::proxy`]) that the
//! consumer sets from its own configuration.

/// Intended cap on any single resolved payload — HTTP, file, or base64.
pub const MAX_FETCH_BYTES: u64 = 64 << 20;

/// Planned byte allowance shared by every source of one request. Charging is
/// not implemented; the stubbed fetch functions currently return an error.
#[derive(Debug)]
pub struct ByteBudget(#[allow(dead_code)] std::sync::atomic::AtomicU64);

impl ByteBudget {
    pub fn new(total: u64) -> Self {
        Self(std::sync::atomic::AtomicU64::new(total))
    }
}

/// Planned knobs of the network stage; [`Default`] matches Python engines' defaults.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct FetchOptions {
    /// Per-source HTTP GET timeout (default 3 s).
    pub timeout: std::time::Duration,
}

impl Default for FetchOptions {
    fn default() -> Self {
        Self {
            timeout: std::time::Duration::from_secs(3),
        }
    }
}

/// Resolve one trusted, string-typed media source into raw encoded bytes.
///
/// Do not call this directly on untrusted request URLs; see the module-level
/// security note.
///
/// # Errors
///
/// Always returns [`MmError::Unsupported`](crate::MmError::Unsupported); not
/// implemented yet.
pub fn fetch_bytes(src: &str) -> crate::Result<Vec<u8>> {
    fetch_bytes_budgeted(src, &ByteBudget::new(MAX_FETCH_BYTES))
}

/// [`fetch_bytes`] against a caller-owned allowance, for resolving several
/// sources under one whole-request bound. [`MAX_FETCH_BYTES`] still caps each.
///
/// # Errors
///
/// Always returns [`MmError::Unsupported`](crate::MmError::Unsupported); not
/// implemented yet.
pub fn fetch_bytes_budgeted(src: &str, budget: &ByteBudget) -> crate::Result<Vec<u8>> {
    fetch_bytes_budgeted_with(src, budget, &FetchOptions::default())
}

/// [`fetch_bytes_budgeted`] with explicit [`FetchOptions`].
///
/// # Errors
///
/// Always returns [`MmError::Unsupported`](crate::MmError::Unsupported); not
/// implemented yet.
pub fn fetch_bytes_budgeted_with(
    src: &str,
    budget: &ByteBudget,
    opts: &FetchOptions,
) -> crate::Result<Vec<u8>> {
    let _ = (src, budget, opts);
    Err(crate::MmError::unsupported(
        "fetch is not implemented yet; resolve media with your own fetcher",
    ))
}

// ---------------------------------------------------------------------------
// Protected fetcher
// ---------------------------------------------------------------------------

use std::collections::HashSet;
use std::net::{IpAddr, Ipv6Addr, SocketAddr};
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use ipnet::IpNet;
use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use reqwest::redirect::Policy;

const DEFAULT_USER_AGENT: &str = concat!("dynamo-multimodal/", env!("CARGO_PKG_VERSION"));
const DEFAULT_HTTP_TIMEOUT: Duration = Duration::from_secs(30);
/// Redirects followed before the next one is refused.
const MAX_REDIRECTS: usize = 3;
/// A `data:` URL carries its whole payload inline, so it gets a tighter default
/// cap than a download.
const DEFAULT_MAX_DATA_URL_BYTES: usize = 16 << 20;
/// Longest `data:` header (`data:image/png;base64`) accepted before the comma.
const MAX_DATA_URL_HEADER: usize = 256;
/// Longest `http(s)` URL accepted, so `Url::parse` never copies an unbounded
/// input (presigned URLs are a few KiB).
const MAX_URL_BYTES: usize = 16 << 10;
/// Largest buffer reserved up front from a declared `Content-Length`; the
/// server's claim is not trusted with more.
const MAX_INITIAL_RESERVE: u64 = 1 << 20;

// Address ranges a user-controlled URL must never reach. The IPv4 and base IPv6
// entries are the list in ai-dynamo/dynamo (`BLOCKED_IP_NETWORKS`) and its
// Python counterpart: RFC 1918 private, RFC 6598 CGNAT, RFC 5735 loopback and
// link-local (169.254/16 covers the AWS / OpenStack metadata address), RFC 4193
// ULA, RFC 6890 reserved. Added here: IPv6 ranges that carry or translate to
// IPv4 and so can reach a private address through a tunnel or translator
// (IPv4-compatible `::/96`, SIIT `::ffff:0:0:0/96`, local-use NAT64
// `64:ff9b:1::/48`, 6to4 `2002::/16`, Teredo `2001::/32`, SRv6 `5f00::/16`),
// and the deprecated, discard, documentation and special-purpose ranges
// (`fec0::/10`, `100::/64`, `2001:10::/28`, `2001:db8::/32`, `192.31.196.0/24`,
// `192.52.193.0/24`, `192.88.99.0/24`). The list is not exhaustive. The well-known
// NAT64 prefix `64:ff9b::/96` is handled in `is_blocked_ip` by checking the
// IPv4 address it embeds.
static BLOCKED_IP_NETWORKS: LazyLock<Vec<IpNet>> = LazyLock::new(|| {
    [
        "0.0.0.0/8",
        "10.0.0.0/8",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.0.0.0/24",
        "192.0.2.0/24",
        "192.31.196.0/24",
        "192.52.193.0/24",
        "192.88.99.0/24",
        "192.168.0.0/16",
        "198.18.0.0/15",
        "198.51.100.0/24",
        "203.0.113.0/24",
        "224.0.0.0/4",
        "240.0.0.0/4",
        "255.255.255.255/32",
        "::/96",
        "::ffff:0:0/96",
        "::ffff:0:0:0/96",
        "64:ff9b:1::/48",
        "100::/64",
        "2001::/32",
        "2001:10::/28",
        "2001:db8::/32",
        "2002::/16",
        "5f00::/16",
        "fc00::/7",
        "fe80::/10",
        "fec0::/10",
        "ff00::/8",
    ]
    .iter()
    .map(|s| s.parse().expect("invalid CIDR in BLOCKED_IP_NETWORKS"))
    .collect()
});

// Hostnames refused by literal match before any DNS lookup, so a hosts-file
// trick or a hostile resolver cannot alias an internal service to a public
// name. Matched case-insensitively, ignoring a trailing dot.
const BLOCKED_HOSTS: [&str; 12] = [
    "localhost",
    "localhost.localdomain",
    "broadcasthost",
    "ip6-allnodes",
    "ip6-allrouters",
    "ip6-localhost",
    "ip6-loopback",
    "metadata",
    "metadata.google.internal",
    "metadata.goog",
    "kubernetes.default",
    "kubernetes.default.svc",
];

// Name suffixes that never name a public host (RFC 6761 `.localhost`, RFC 6762
// `.local`, and the conventional `.internal` and `.localdomain`).
const BLOCKED_SUFFIXES: [&str; 4] = [".localhost", ".localdomain", ".local", ".internal"];

/// `true` if `ip` is inside a blocked range, including an IPv4 address embedded
/// in a well-known NAT64 (`64:ff9b::/96`) address.
pub fn is_blocked_ip(ip: &IpAddr) -> bool {
    if BLOCKED_IP_NETWORKS.iter().any(|net| net.contains(ip)) {
        return true;
    }
    if let IpAddr::V6(v6) = ip
        && is_well_known_nat64(v6)
    {
        let [.., a, b, c, d] = v6.octets();
        return is_blocked_ip(&IpAddr::from([a, b, c, d]));
    }
    false
}

fn is_well_known_nat64(ip: &Ipv6Addr) -> bool {
    ip.octets()[..12] == [0, 0x64, 0xff, 0x9b, 0, 0, 0, 0, 0, 0, 0, 0]
}

/// A policy refusal. A distinct type so it can ride through reqwest's resolver
/// and redirect hooks and still be recognised afterwards, when reqwest has
/// wrapped it in its own error.
#[derive(Debug)]
struct Rejected(String);

impl std::fmt::Display for Rejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Rejected {}

impl From<Rejected> for crate::MmError {
    fn from(r: Rejected) -> Self {
        crate::MmError::invalid_input(r.0)
    }
}

fn rejected(message: impl Into<String>) -> Rejected {
    Rejected(message.into())
}

/// What a [`MediaFetcher`] may reach and how much it may read. Plain data: set
/// the fields, then build the fetcher; changing a policy afterwards does not
/// affect a fetcher that already exists.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct FetchPolicy {
    pub user_agent: String,
    /// Allow an IP-literal host (`http://203.0.113.5/x`).
    pub allow_direct_ip: bool,
    /// Allow an explicit non-default port (`http://host:8080/x`). A URL that
    /// spells the scheme's default port (`:80` on http, `:443` on https) is
    /// normalised to none by the URL parser and is not affected.
    pub allow_direct_port: bool,
    /// When `false` (the default), refuse blocked hostnames and name suffixes,
    /// IP literals in a blocked range, and hostnames that resolve to one. This
    /// is one switch for "allow internal / on-prem targets": private ranges and
    /// internal service names are needed together on-prem. Never enable it on a
    /// public-facing service.
    pub allow_private_ips: bool,
    /// If set, only these hostnames may be fetched. Compared case-insensitively
    /// and ignoring a trailing dot; internationalised names must be given in
    /// their punycode form.
    pub allowed_media_domains: Option<HashSet<String>>,
    /// Whole-fetch timeout, DNS pre-flight included. `None` disables it. The
    /// system resolver runs on Tokio's blocking pool, and a lookup already in
    /// flight keeps running after the timeout fires; a stalled resolver can
    /// therefore still occupy blocking threads. A `data:` URL involves no
    /// waiting and is not bounded by it; see `max_data_url_bytes`.
    pub timeout: Option<Duration>,
    /// Cap on a downloaded body, in bytes.
    pub max_bytes: u64,
    /// Cap on a `data:` URL's payload, in bytes. The payload is decoded inline
    /// on the calling task, so this cap is what bounds that work.
    pub max_data_url_bytes: usize,
    /// Send every request through this `http`/`https` proxy, e.g.
    /// `http://proxy.corp:3128`. `None` (the default) connects directly.
    ///
    /// Every request goes through the proxy, with no `NO_PROXY` exceptions, so
    /// nothing connects directly without the DNS filter. The proxy resolves
    /// destinations, so [`MediaFetcher::fetch`] skips its local DNS check (it
    /// would refuse hosts only the proxy can resolve); the URL policy checks
    /// still apply. Only the proxy's own hostname is resolved without the
    /// private-range filter, since a proxy usually sits on a private address.
    /// Set it only for a proxy you trust to enforce its own destination
    /// policy. The proxy environment variables are never read.
    pub proxy: Option<String>,
}

impl Default for FetchPolicy {
    fn default() -> Self {
        Self {
            user_agent: DEFAULT_USER_AGENT.to_string(),
            allow_direct_ip: false,
            allow_direct_port: false,
            allow_private_ips: false,
            allowed_media_domains: None,
            timeout: Some(DEFAULT_HTTP_TIMEOUT),
            max_bytes: MAX_FETCH_BYTES,
            max_data_url_bytes: DEFAULT_MAX_DATA_URL_BYTES,
            proxy: None,
        }
    }
}

impl FetchPolicy {
    /// The default policy, or the on-prem opt-in: `allow_internal` flips
    /// `allow_direct_ip`, `allow_direct_port` and `allow_private_ips` together.
    pub fn with_internal_access(allow_internal: bool) -> Self {
        Self {
            allow_direct_ip: allow_internal,
            allow_direct_port: allow_internal,
            allow_private_ips: allow_internal,
            ..Self::default()
        }
    }

    fn check(&self, url: &url::Url) -> std::result::Result<(), Rejected> {
        if !matches!(url.scheme(), "http" | "https" | "data") {
            return Err(rejected("Only HTTP(S) and data URLs are allowed"));
        }
        if url.scheme() == "data" {
            return Ok(());
        }
        if !url.username().is_empty() || url.password().is_some() {
            return Err(rejected("URLs with embedded credentials are not allowed"));
        }
        let host = url
            .host()
            .ok_or_else(|| rejected("URL has no host component"))?;
        if !self.allow_direct_ip && !matches!(host, url::Host::Domain(_)) {
            return Err(rejected("Direct IP access is not allowed"));
        }
        if !self.allow_direct_port && url.port().is_some() {
            return Err(rejected("Direct port access is not allowed"));
        }
        if !self.allow_private_ips {
            let ip_literal = match host {
                url::Host::Domain(domain) => {
                    let lowered = domain.trim_end_matches('.').to_ascii_lowercase();
                    if BLOCKED_HOSTS.contains(&lowered.as_str())
                        || BLOCKED_SUFFIXES.iter().any(|s| lowered.ends_with(s))
                    {
                        return Err(rejected(format!(
                            "Host '{domain}' is blocked (internal name)"
                        )));
                    }
                    None
                }
                url::Host::Ipv4(ip) => Some(IpAddr::V4(ip)),
                url::Host::Ipv6(ip) => Some(IpAddr::V6(ip)),
            };
            if let Some(ip) = ip_literal
                && is_blocked_ip(&ip)
            {
                return Err(rejected(format!("IP literal '{ip}' is in a blocked range")));
            }
        }
        if let Some(allowed) = &self.allowed_media_domains
            && let Some(host_str) = url.host_str()
        {
            let host = host_str.trim_end_matches('.');
            if !allowed
                .iter()
                .any(|d| d.trim_end_matches('.').eq_ignore_ascii_case(host))
            {
                return Err(rejected(format!(
                    "Host '{host_str}' is not in the allowed_media_domains list"
                )));
            }
        }
        Ok(())
    }

    /// Whether a redirect to `next` may be followed. `history_len` is reqwest's
    /// `previous().len()`: the original URL plus the hops followed so far.
    fn check_redirect(
        &self,
        history_len: usize,
        next: &url::Url,
    ) -> std::result::Result<(), Rejected> {
        if history_len > MAX_REDIRECTS {
            return Err(rejected(format!(
                "too many redirects (max={MAX_REDIRECTS})"
            )));
        }
        // A redirect can only continue over HTTP; a `data:` target would never
        // reach the decoder and is a refusal, not a transport error.
        if !matches!(next.scheme(), "http" | "https") {
            return Err(rejected("Redirects must stay on HTTP(S)"));
        }
        self.check(next)
    }
}

/// Fetches untrusted `http`, `https` and `data:` media URLs under a
/// [`FetchPolicy`].
///
/// Policy refusals are [`MmError::InvalidInput`](crate::MmError::InvalidInput);
/// an over-cap payload is [`MmError::LimitExceeded`](crate::MmError::LimitExceeded);
/// transport failures, timeouts and HTTP 5xx are
/// [`MmError::Internal`](crate::MmError::Internal).
///
/// Build one fetcher and reuse it: the connection pool stays warm, and the
/// policy it was built with is applied to the request, to every redirect hop
/// (revalidated and capped), and to DNS answers (blocked addresses are dropped
/// before reqwest sees them, so rebinding cannot reach one). The pre-flight DNS
/// check that gives an early, clear refusal resolves the name once more than the
/// connection does.
///
/// Blocking is CPU-light but async: call [`fetch`](Self::fetch) from a Tokio
/// runtime.
#[derive(Clone, Debug)]
pub struct MediaFetcher {
    policy: FetchPolicy,
    client: reqwest::Client,
}

impl MediaFetcher {
    /// Build a fetcher that enforces `policy`.
    pub fn new(policy: FetchPolicy) -> crate::Result<Self> {
        let client = client_builder(&policy)?.build().map_err(|e| {
            crate::MmError::internal_with_source("could not build the http client", e)
        })?;
        Ok(Self { policy, client })
    }

    /// The policy this fetcher enforces.
    pub fn policy(&self) -> &FetchPolicy {
        &self.policy
    }

    /// Synchronous policy check of one URL: scheme, credentials, host form,
    /// port, blocked hostnames and IP literals, domain allowlist. Does not
    /// resolve DNS.
    pub fn check_url(&self, url: &str) -> crate::Result<()> {
        match Source::read(url)? {
            Source::Data(_) => Ok(()),
            Source::Url(url) => Ok(self.policy.check(&url)?),
        }
    }

    /// [`check_url`](Self::check_url), then, for hostnames, resolve DNS and
    /// refuse if any answer is in a blocked range. A lookup failure is refused
    /// too.
    pub async fn check_url_with_dns(&self, url: &str) -> crate::Result<()> {
        match Source::read(url)? {
            Source::Data(_) => Ok(()),
            Source::Url(url) => self.preflight(&url).await,
        }
    }

    async fn preflight(&self, url: &url::Url) -> crate::Result<()> {
        self.policy.check(url)?;
        if self.policy.allow_private_ips || url.scheme() == "data" {
            return Ok(());
        }
        let Some(url::Host::Domain(host)) = url.host() else {
            return Ok(());
        };
        let port = url.port_or_known_default().unwrap_or(0);
        let answers = tokio::net::lookup_host((host, port)).await.map_err(|e| {
            crate::MmError::internal_with_source(format!("could not resolve host '{host}'"), e)
        })?;
        for answer in answers {
            let ip = answer.ip();
            if is_blocked_ip(&ip) {
                return Err(
                    rejected(format!("Host '{host}' resolves to blocked IP '{ip}'")).into(),
                );
            }
        }
        Ok(())
    }

    /// Fetch one media URL into raw bytes.
    ///
    /// `data:` URLs must be base64 and are decoded inline, up to
    /// `max_data_url_bytes`; `http(s)` bodies are streamed and stop at
    /// `max_bytes`. An empty payload is an error. An `http(s)` fetch, DNS
    /// pre-flight included, is bounded by `timeout`.
    pub async fn fetch(&self, src: &str) -> crate::Result<Vec<u8>> {
        match self.policy.timeout {
            Some(limit) => tokio::time::timeout(limit, self.fetch_inner(src))
                .await
                .map_err(|_| crate::MmError::internal("media fetch timed out"))?,
            None => self.fetch_inner(src).await,
        }
    }

    async fn fetch_inner(&self, src: &str) -> crate::Result<Vec<u8>> {
        let url = match Source::read(src)? {
            Source::Data(src) => return self.decode_data_url(src),
            Source::Url(url) => url,
        };
        // Behind a proxy the proxy resolves the destination; a local lookup
        // would refuse hosts only it can resolve.
        if self.policy.proxy.is_some() {
            self.policy.check(&url)?;
        } else {
            self.preflight(&url).await?;
        }
        self.download(&url).await
    }

    fn decode_data_url(&self, src: &str) -> crate::Result<Vec<u8>> {
        // Look for the comma only within the allowed header, so a long run with
        // no comma costs a bounded scan.
        let window = &src.as_bytes()[..src.len().min(MAX_DATA_URL_HEADER + 1)];
        let (header, payload) = window
            .iter()
            .position(|&b| b == b',')
            .map(|comma| (&src[..comma], &src[comma + 1..]))
            .ok_or_else(|| crate::MmError::invalid_input("invalid media data URL"))?;
        if !header.ends_with(";base64") && !header.to_ascii_lowercase().ends_with(";base64") {
            return Err(crate::MmError::invalid_input(
                "media data URLs must be base64-encoded",
            ));
        }
        if payload.len() > self.policy.max_data_url_bytes {
            return Err(crate::MmError::limit_exceeded(format!(
                "data URL payload of {} bytes exceeds the {} byte limit",
                payload.len(),
                self.policy.max_data_url_bytes
            )));
        }
        let bytes = BASE64.decode(payload).map_err(|e| {
            crate::MmError::invalid_input_with_source("invalid base64 in data URL", e)
        })?;
        if bytes.is_empty() {
            return Err(crate::MmError::invalid_input("media data URL is empty"));
        }
        Ok(bytes)
    }

    async fn download(&self, url: &url::Url) -> crate::Result<Vec<u8>> {
        let mut response = self
            .client
            .get(url.clone())
            .send()
            .await
            .map_err(map_reqwest)?;
        let status = response.status();
        if !status.is_success() {
            let message = format!("media url returned HTTP {status}");
            // A 3xx here is a redirect reqwest declined to follow (a target it
            // cannot request); like a 4xx it is about the URL, not our server.
            return Err(if status.is_client_error() || status.is_redirection() {
                crate::MmError::invalid_input(message)
            } else {
                crate::MmError::internal(message)
            });
        }
        let cap = self.policy.max_bytes;
        let declared = response.content_length();
        if declared.is_some_and(|n| n > cap) {
            return Err(over_cap(cap));
        }
        // The declared length is the server's claim, already within the cap but
        // not trusted with a large up-front allocation.
        let reserve = declared.unwrap_or(0).min(MAX_INITIAL_RESERVE);
        let mut body = Vec::with_capacity(usize::try_from(reserve).unwrap_or(0));
        while let Some(chunk) = response.chunk().await.map_err(map_reqwest)? {
            if (body.len() as u64).saturating_add(chunk.len() as u64) > cap {
                return Err(over_cap(cap));
            }
            body.extend_from_slice(&chunk);
        }
        if body.is_empty() {
            return Err(crate::MmError::invalid_input(
                "media url returned an empty body",
            ));
        }
        Ok(body)
    }
}

/// A media URL read without copying a payload of any size.
enum Source<'a> {
    /// A `data:` URL, checked and decoded on the borrowed string.
    Data(&'a str),
    Url(url::Url),
}

impl<'a> Source<'a> {
    fn read(src: &'a str) -> crate::Result<Self> {
        // Like `Url::parse`, ignore leading and trailing C0 controls and spaces.
        let src = src.trim_matches(|c: char| c <= ' ');
        if src
            .get(..5)
            .is_some_and(|scheme| scheme.eq_ignore_ascii_case("data:"))
        {
            return Ok(Self::Data(src));
        }
        // Refuse before `Url::parse` copies the input.
        if src.len() > MAX_URL_BYTES {
            return Err(crate::MmError::invalid_input(format!(
                "media url is longer than {MAX_URL_BYTES} bytes"
            )));
        }
        let url = url::Url::parse(src)
            .map_err(|e| crate::MmError::invalid_input_with_source("invalid media url", e))?;
        // `Url::parse` ignores tabs and newlines inside the scheme, so `da\tta:`
        // gets here as a data URL; only the borrowed-string path decodes those,
        // so refuse it rather than hand it to the HTTP client.
        if url.scheme() == "data" {
            return Err(crate::MmError::invalid_input(
                "data URLs must start with \"data:\"",
            ));
        }
        Ok(Self::Url(url))
    }
}

fn over_cap(cap: u64) -> crate::MmError {
    crate::MmError::limit_exceeded(format!("media download exceeds the {cap} byte limit"))
}

/// The reqwest client that enforces `policy`: redirect revalidation, a DNS
/// resolver that drops blocked addresses, the user agent, the request timeout
/// and the policy's proxy (system proxies are never used). No `Referer` is
/// sent on redirects: reqwest's would carry the previous URL's query string,
/// which can hold a presigned token, to the redirect target.
fn client_builder(policy: &FetchPolicy) -> crate::Result<reqwest::ClientBuilder> {
    let proxy = policy.proxy.as_deref().map(parse_proxy).transpose()?;
    let for_redirects = policy.clone();
    let redirects = Policy::custom(move |attempt| {
        match for_redirects.check_redirect(attempt.previous().len(), attempt.url()) {
            Ok(()) => attempt.follow(),
            Err(e) => attempt.error(e),
        }
    });
    let mut builder = reqwest::Client::builder()
        .user_agent(&policy.user_agent)
        .redirect(redirects)
        .referer(false)
        .dns_resolver(Arc::new(BlocklistResolver {
            allow_private_ips: policy.allow_private_ips,
            proxy_host: proxy.as_ref().map(|(host, _)| host.clone()),
        }));
    if let Some(timeout) = policy.timeout {
        builder = builder.timeout(timeout);
    }
    // An explicit proxy also turns off reqwest's environment lookup, and
    // `Proxy::all` has no `NO_PROXY` exceptions.
    Ok(match proxy {
        Some((_, proxy)) => builder.proxy(proxy),
        None => builder.no_proxy(),
    })
}

/// The policy's proxy URL and its hostname, lowercased without a trailing dot.
/// Errors leave the URL out: it can carry proxy credentials.
fn parse_proxy(proxy: &str) -> crate::Result<(String, reqwest::Proxy)> {
    let invalid =
        || crate::MmError::invalid_input("the fetch policy's proxy is not a valid http(s) URL");
    let url = url::Url::parse(proxy).map_err(|_| invalid())?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(invalid());
    }
    let host = url
        .host_str()
        .ok_or_else(invalid)?
        .trim_end_matches('.')
        .to_ascii_lowercase();
    let proxy = reqwest::Proxy::all(proxy).map_err(|_| invalid())?;
    Ok((host, proxy))
}

/// Turn a reqwest failure into the crate's taxonomy, recovering a policy
/// refusal that reqwest wrapped (resolver and redirect hooks). The retained
/// error has its URL removed: it can carry query-string tokens.
fn map_reqwest(error: reqwest::Error) -> crate::MmError {
    let error = error.without_url();
    let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(&error);
    while let Some(e) = cause {
        if let Some(r) = e.downcast_ref::<Rejected>() {
            return crate::MmError::invalid_input(r.0.clone());
        }
        cause = e.source();
    }
    if error.is_redirect() {
        // reqwest uses this category for malformed redirects too, so do not
        // claim policy was necessarily the cause.
        return crate::MmError::invalid_input_with_source(
            "media url redirect was not followed (blocked destination, redirect limit, or invalid redirect)",
            error,
        );
    }
    crate::MmError::internal_with_source("media fetch failed", error)
}

/// The addresses a hostname may be connected to: every answer when private
/// access is allowed, otherwise only the unblocked ones. No answer left is a
/// refusal.
fn filter_resolved(
    host: &str,
    answers: impl Iterator<Item = SocketAddr>,
    allow_private: bool,
) -> std::result::Result<Vec<SocketAddr>, Rejected> {
    let addrs: Vec<SocketAddr> = if allow_private {
        answers.collect()
    } else {
        answers.filter(|sa| !is_blocked_ip(&sa.ip())).collect()
    };
    if addrs.is_empty() {
        return Err(rejected(format!(
            "no non-blocked addresses for host '{host}'"
        )));
    }
    Ok(addrs)
}

async fn resolve_filtered(
    host: &str,
    allow_private: bool,
) -> Result<Vec<SocketAddr>, Box<dyn std::error::Error + Send + Sync>> {
    let answers = tokio::net::lookup_host((host, 0_u16)).await?;
    Ok(filter_resolved(host, answers, allow_private)?)
}

/// DNS resolver that drops blocked addresses before reqwest sees them.
/// reqwest calls it for every hostname it connects to, redirect targets
/// included, so DNS rebinding cannot slip a blocked address past the policy.
/// Behind a proxy it only ever resolves the proxy, which may be private.
struct BlocklistResolver {
    allow_private_ips: bool,
    proxy_host: Option<String>,
}

impl BlocklistResolver {
    fn allows_private(&self, host: &str) -> bool {
        self.allow_private_ips
            || self
                .proxy_host
                .as_deref()
                .is_some_and(|proxy| host.trim_end_matches('.').eq_ignore_ascii_case(proxy))
    }
}

impl Resolve for BlocklistResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let host = name.as_str().to_string();
        let allow_private = self.allows_private(&host);
        Box::pin(async move {
            let addrs = resolve_filtered(&host, allow_private).await?;
            Ok(Box::new(addrs.into_iter()) as Addrs)
        })
    }
}

#[cfg(test)]
mod fetcher_tests {
    use super::*;
    use crate::MmError;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    fn url(s: &str) -> url::Url {
        url::Url::parse(s).unwrap()
    }

    fn refused(r: std::result::Result<(), Rejected>) -> bool {
        r.is_err()
    }

    fn fetcher(policy: FetchPolicy) -> MediaFetcher {
        MediaFetcher::new(policy).unwrap()
    }

    fn internal() -> MediaFetcher {
        fetcher(FetchPolicy::with_internal_access(true))
    }

    /// Serve `handler(request_path)` as the raw bytes of an HTTP response to
    /// every connection on a loopback port; the counter records connections.
    async fn serve(
        handler: impl Fn(&str) -> Vec<u8> + Send + Sync + 'static,
    ) -> (SocketAddr, Arc<AtomicUsize>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handler = Arc::new(handler);
        let connections = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&connections);
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                counted.fetch_add(1, Ordering::SeqCst);
                let handler = Arc::clone(&handler);
                tokio::spawn(async move {
                    let mut request = Vec::new();
                    let mut buf = [0u8; 1024];
                    while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                        match socket.read(&mut buf).await {
                            Ok(0) | Err(_) => return,
                            Ok(n) => request.extend_from_slice(&buf[..n]),
                        }
                    }
                    let line = String::from_utf8_lossy(&request);
                    let path = line.split_whitespace().nth(1).unwrap_or("/").to_string();
                    let _ = socket.write_all(&handler(&path)).await;
                    let _ = socket.shutdown().await;
                });
            }
        });
        (addr, connections)
    }

    fn ok(body: &[u8]) -> Vec<u8> {
        let mut r = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .into_bytes();
        r.extend_from_slice(body);
        r
    }

    fn status(code: &str) -> Vec<u8> {
        format!("HTTP/1.1 {code}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").into_bytes()
    }

    fn redirect(to: &str) -> Vec<u8> {
        format!(
            "HTTP/1.1 302 Found\r\nLocation: {to}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        )
        .into_bytes()
    }

    #[test]
    fn only_http_https_and_data_schemes_pass() {
        let p = FetchPolicy::default();
        for u in [
            "http://example.com/a.jpg",
            "https://example.com/a.jpg",
            "data:image/png;base64,AAAA",
        ] {
            assert!(p.check(&url(u)).is_ok(), "{u}");
        }
        for u in [
            "ftp://example.com/a",
            "file:///etc/passwd",
            "gopher://example.com/",
        ] {
            assert!(refused(p.check(&url(u))), "{u}");
        }
    }

    #[test]
    fn embedded_credentials_are_refused() {
        let p = FetchPolicy::default();
        for u in [
            "http://user:pw@example.com/a",
            "http://user@example.com/a",
            "http://good.com@169.254.169.254/",
        ] {
            assert!(refused(p.check(&url(u))), "{u}");
        }
    }

    #[test]
    fn direct_ip_and_port_are_refused_by_default() {
        let p = FetchPolicy::default();
        assert!(refused(p.check(&url("http://203.0.113.5/a"))));
        assert!(refused(p.check(&url("http://[2606:4700::1111]/a"))));
        assert!(refused(p.check(&url("http://example.com:8080/a"))));
        let direct = FetchPolicy {
            allow_direct_ip: true,
            allow_direct_port: true,
            ..FetchPolicy::default()
        };
        assert!(direct.check(&url("http://8.8.8.8:8080/a")).is_ok());
        assert!(refused(direct.check(&url("http://10.0.0.1/a"))));
        assert!(refused(
            direct.check(&url("http://169.254.169.254/latest/meta-data"))
        ));
        assert!(refused(direct.check(&url("http://[::1]/a"))));
    }

    #[test]
    fn numeric_spellings_of_loopback_are_normalised_then_refused() {
        let p = FetchPolicy {
            allow_direct_ip: true,
            ..FetchPolicy::default()
        };
        for u in [
            "http://2130706433/",
            "http://0x7f.1/",
            "http://127.1/",
            "http://0177.0.0.1/",
            "http://[::ffff:7f00:1]/",
            "http://[::ffff:127.0.0.1]/",
        ] {
            assert!(refused(p.check(&url(u))), "{u}");
        }
    }

    #[test]
    fn blocked_hostnames_and_suffixes_are_refused_however_spelled() {
        let p = FetchPolicy::default();
        for u in [
            "http://localhost/a",
            "http://LOCALHOST/a",
            "http://localhost./a",
            "http://metadata.google.internal/computeMetadata/v1/",
            "http://kubernetes.default.svc/a",
            "http://app.localhost/a",
            "http://printer.local/a",
            "http://db.corp.internal/a",
            "http://host.localdomain/a",
            "http://broadcasthost/a",
        ] {
            assert!(refused(p.check(&url(u))), "{u}");
        }
        let on_prem = FetchPolicy::with_internal_access(true);
        assert!(on_prem.check(&url("http://localhost/a")).is_ok());
        assert!(on_prem.check(&url("http://db.corp.internal/a")).is_ok());
    }

    #[test]
    fn blocked_ranges() {
        for ip in [
            "10.1.2.3",
            "172.16.0.1",
            "192.168.1.1",
            "127.0.0.1",
            "169.254.169.254",
            "100.64.0.1",
            "0.0.0.0",
            "224.0.0.1",
            "255.255.255.255",
            "192.88.99.1",
            "192.31.196.1",
            "192.52.193.1",
            "::1",
            "::",
            "fc00::1",
            "fe80::1",
            "::ffff:127.0.0.1",
        ] {
            assert!(is_blocked_ip(&ip.parse().unwrap()), "{ip}");
        }
        for ip in [
            "8.8.8.8",
            "1.1.1.1",
            "93.184.216.34",
            "172.32.0.1",
            "2606:4700::1111",
        ] {
            assert!(!is_blocked_ip(&ip.parse().unwrap()), "{ip}");
        }
    }

    #[test]
    fn ipv6_forms_that_reach_ipv4_are_blocked() {
        for ip in [
            "64:ff9b::7f00:1",                      // NAT64 to 127.0.0.1
            "64:ff9b::a00:1",                       // NAT64 to 10.0.0.1
            "64:ff9b::a9fe:a9fe",                   // NAT64 to 169.254.169.254
            "64:ff9b:1::a00:1",                     // local-use NAT64
            "2002:7f00:1::",                        // 6to4 of 127.0.0.1
            "2002:a00:1::1",                        // 6to4 of 10.0.0.1
            "2001:0:4136:e378:8000:63bf:3fff:fdd2", // Teredo
            "::7f00:1",                             // IPv4-compatible
            "fec0::1",                              // site-local
            "100::1",                               // discard
            "2001:db8::1",                          // documentation
            "::ffff:0:7f00:1",                      // SIIT of 127.0.0.1
            "2001:10::1",                           // ORCHID
            "5f00::1",                              // SRv6
        ] {
            assert!(is_blocked_ip(&ip.parse().unwrap()), "{ip}");
        }
        // NAT64 to a public address stays reachable.
        assert!(!is_blocked_ip(&"64:ff9b::808:808".parse().unwrap()));
    }

    #[test]
    fn domain_allowlist_ignores_case_and_trailing_dot() {
        let p = FetchPolicy {
            allowed_media_domains: Some(["CDN.Example.com".to_string()].into()),
            ..FetchPolicy::default()
        };
        assert!(p.check(&url("https://cdn.example.com/a.jpg")).is_ok());
        assert!(p.check(&url("https://cdn.example.com./a.jpg")).is_ok());
        assert!(refused(p.check(&url("https://evil.example.org/a.jpg"))));
        // data: URLs carry no host and are not subject to the allowlist.
        assert!(p.check(&url("data:image/png;base64,AAAA")).is_ok());
    }

    #[test]
    fn redirect_history_counts_the_original_url() {
        let p = FetchPolicy {
            allow_direct_ip: true,
            ..FetchPolicy::default()
        };
        let public = url("http://8.8.8.8/a");
        // reqwest's `previous()` is the original plus the hops followed: with
        // MAX_REDIRECTS = 3, the third redirect arrives with a history of 3.
        assert!(p.check_redirect(1, &public).is_ok());
        assert!(p.check_redirect(MAX_REDIRECTS, &public).is_ok());
        assert!(p.check_redirect(MAX_REDIRECTS + 1, &public).is_err());
        for blocked in [
            "http://169.254.169.254/x",
            "http://localhost/x",
            "ftp://example.com/x",
            "file:///etc/passwd",
            "data:image/png;base64,AAAA",
        ] {
            assert!(p.check_redirect(1, &url(blocked)).is_err(), "{blocked}");
        }
    }

    #[test]
    fn dns_answers_in_blocked_ranges_are_dropped() {
        let sa = |s: &str| -> SocketAddr { format!("{s}:0").parse().unwrap() };
        let mixed = [sa("10.0.0.5"), sa("93.184.216.34")];
        let kept = filter_resolved("h", mixed.iter().copied(), false).unwrap();
        assert_eq!(kept, vec![sa("93.184.216.34")]);
        // Every answer blocked: refuse rather than fall back to one.
        assert!(filter_resolved("h", [sa("127.0.0.1")].into_iter(), false).is_err());
        // Private access allowed: nothing is dropped.
        assert_eq!(
            filter_resolved("h", mixed.iter().copied(), true)
                .unwrap()
                .len(),
            2
        );
    }

    #[tokio::test]
    async fn resolver_refuses_a_name_that_resolves_to_loopback() {
        // `localhost` resolves to loopback on every platform; this is the code
        // path reqwest's resolver hook runs for each connection.
        let err = resolve_filtered("localhost", false).await.unwrap_err();
        assert!(err.downcast_ref::<Rejected>().is_some(), "{err}");
        assert!(
            !resolve_filtered("localhost", true)
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn data_urls_decode_and_are_bounded() {
        let f = fetcher(FetchPolicy::default());
        assert_eq!(
            f.fetch("data:image/png;base64,dGVzdA==").await.unwrap(),
            b"test"
        );
        // Leading and trailing spaces are ignored, as `Url::parse` ignores them.
        assert_eq!(
            f.fetch("  DATA:image/png;BASE64,dGVzdA==\n").await.unwrap(),
            b"test"
        );
        let too_long_header = format!("data:{};base64,dGVzdA==", "x".repeat(MAX_DATA_URL_HEADER));
        for bad in [
            "data:image/png;base64,",     // empty payload
            "data:image/png,hello",       // not base64
            "data:image/png;base64,!!!!", // invalid base64
            "data:image/png;base64",      // no comma
            too_long_header.as_str(),     // header far past the limit
        ] {
            let r = f.fetch(bad).await;
            assert!(
                matches!(r, Err(MmError::InvalidInput { .. })),
                "{bad:.40}: {r:?}"
            );
        }
        let small = fetcher(FetchPolicy {
            max_data_url_bytes: 4,
            ..FetchPolicy::default()
        });
        let r = small.fetch("data:image/png;base64,dGVzdA==").await;
        assert!(matches!(r, Err(MmError::LimitExceeded { .. })), "{r:?}");
    }

    #[tokio::test]
    async fn fetches_over_http_and_maps_failures() {
        let (addr, _) = serve(|path| match path {
            "/ok" => ok(b"payload"),
            "/missing" => status("404 Not Found"),
            "/boom" => status("500 Internal Server Error"),
            "/empty" => ok(b""),
            _ => status("400 Bad Request"),
        })
        .await;
        let f = internal();
        let get = |p: &str| format!("http://{addr}{p}");
        assert_eq!(f.fetch(&get("/ok")).await.unwrap(), b"payload");
        let r = f.fetch(&get("/missing")).await;
        assert!(matches!(r, Err(MmError::InvalidInput { .. })), "404: {r:?}");
        let r = f.fetch(&get("/boom")).await;
        assert!(matches!(r, Err(MmError::Internal { .. })), "500: {r:?}");
        let r = f.fetch(&get("/empty")).await;
        assert!(
            matches!(r, Err(MmError::InvalidInput { .. })),
            "empty: {r:?}"
        );
    }

    #[tokio::test]
    async fn policy_refusal_makes_no_connection() {
        let (addr, connections) = serve(|_| ok(b"secret")).await;
        let f = fetcher(FetchPolicy::default());
        let r = f.fetch(&format!("http://{addr}/ok")).await;
        assert!(matches!(r, Err(MmError::InvalidInput { .. })), "{r:?}");
        let r = f.fetch("http://user:pw@example.com/").await;
        assert!(matches!(r, Err(MmError::InvalidInput { .. })), "{r:?}");
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(connections.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn body_over_the_cap_stops_the_download() {
        let big = vec![7u8; 4096];
        let declared = big.clone();
        let (addr, _) = serve(move |path| {
            if path == "/declared" {
                ok(&declared)
            } else {
                // No Content-Length: the cap must still stop the read.
                let mut r = b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n".to_vec();
                r.extend_from_slice(&declared);
                r
            }
        })
        .await;
        let f = fetcher(FetchPolicy {
            max_bytes: 1024,
            ..FetchPolicy::with_internal_access(true)
        });
        for p in ["/declared", "/streamed"] {
            let r = f.fetch(&format!("http://{addr}{p}")).await;
            assert!(
                matches!(r, Err(MmError::LimitExceeded { .. })),
                "{p}: {r:?}"
            );
        }
        let roomy = fetcher(FetchPolicy {
            max_bytes: 4096,
            ..FetchPolicy::with_internal_access(true)
        });
        assert_eq!(
            roomy
                .fetch(&format!("http://{addr}/declared"))
                .await
                .unwrap(),
            big
        );
    }

    #[tokio::test]
    async fn a_server_that_never_answers_hits_the_timeout() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let mut held = Vec::new();
            while let Ok((sock, _)) = listener.accept().await {
                held.push(sock); // accept, then say nothing
            }
        });
        let f = fetcher(FetchPolicy {
            timeout: Some(Duration::from_millis(200)),
            ..FetchPolicy::with_internal_access(true)
        });
        let started = std::time::Instant::now();
        let r = f.fetch(&format!("http://{addr}/slow")).await;
        assert!(matches!(r, Err(MmError::Internal { .. })), "{r:?}");
        let waited = started.elapsed();
        assert!(
            waited >= Duration::from_millis(150),
            "failed early after {waited:?}"
        );
        assert!(waited < Duration::from_secs(5), "{waited:?}");
    }

    #[tokio::test]
    async fn redirects_follow_three_hops_and_refuse_a_fourth() {
        let (addr, _) = serve(|path| match path {
            "/a3" => redirect("/a2"),
            "/a2" => redirect("/a1"),
            "/a1" => redirect("/done"),
            "/b4" => redirect("/a3"),
            "/done" => ok(b"arrived"),
            _ => status("404 Not Found"),
        })
        .await;
        let f = internal();
        assert_eq!(
            f.fetch(&format!("http://{addr}/a3")).await.unwrap(),
            b"arrived"
        );
        let r = f.fetch(&format!("http://{addr}/b4")).await;
        assert!(matches!(r, Err(MmError::InvalidInput { .. })), "{r:?}");
        assert!(format!("{r:?}").contains("too many redirects"), "{r:?}");
    }

    #[tokio::test]
    async fn a_proxy_on_a_private_address_carries_every_request() {
        // The proxy records the request target it is asked for.
        let asked = Arc::new(Mutex::new(Vec::<String>::new()));
        let record = Arc::clone(&asked);
        let (addr, _) = serve(move |target| {
            record.lock().unwrap().push(target.to_string());
            ok(b"via proxy")
        })
        .await;
        // A default policy: private addresses refused, yet the proxy at
        // localhost is reached, and the target (a `.test` name, which no
        // resolver answers) is resolved by the proxy, not locally.
        let f = fetcher(FetchPolicy {
            proxy: Some(format!("http://localhost:{}", addr.port())),
            ..FetchPolicy::default()
        });
        let r = f.fetch("http://media.example.test/clip.mp4").await;
        assert_eq!(r.unwrap(), b"via proxy");
        assert_eq!(
            *asked.lock().unwrap(),
            ["http://media.example.test/clip.mp4"]
        );
    }

    #[tokio::test]
    async fn behind_a_proxy_the_url_policy_still_applies() {
        let (addr, connections) = serve(|_| ok(b"via proxy")).await;
        let f = fetcher(FetchPolicy {
            proxy: Some(format!("http://localhost:{}", addr.port())),
            ..FetchPolicy::default()
        });
        for blocked in ["http://localhost/x", "http://169.254.169.254/x"] {
            let r = f.fetch(blocked).await;
            assert!(
                matches!(r, Err(MmError::InvalidInput { .. })),
                "{blocked}: {r:?}"
            );
        }
        assert_eq!(connections.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn only_the_proxy_host_may_resolve_to_a_private_address() {
        let resolver = BlocklistResolver {
            allow_private_ips: false,
            proxy_host: Some("proxy.corp".to_string()),
        };
        assert!(resolver.allows_private("proxy.corp"));
        assert!(resolver.allows_private("PROXY.corp."));
        assert!(!resolver.allows_private("other.corp"));
        assert!(!resolver.allows_private("proxy.corp.evil.example"));
        let direct = BlocklistResolver {
            allow_private_ips: false,
            proxy_host: None,
        };
        assert!(!direct.allows_private("proxy.corp"));
    }

    #[test]
    fn an_invalid_proxy_is_refused_without_echoing_it() {
        for proxy in [
            "not a url",
            "ftp://user:SENTINEL_SECRET@proxy.corp",
            "http://",
        ] {
            let r = MediaFetcher::new(FetchPolicy {
                proxy: Some(proxy.to_string()),
                ..FetchPolicy::default()
            });
            let Err(err) = r else {
                panic!("{proxy} was accepted");
            };
            assert!(matches!(err, MmError::InvalidInput { .. }), "{err:?}");
            assert!(!format!("{err} {err:?}").contains("SENTINEL_SECRET"));
        }
    }

    #[tokio::test]
    async fn redirects_do_not_send_a_referer() {
        // The redirect target, on another host name, records each request.
        let target = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = target.local_addr().unwrap().port();
        let seen = Arc::new(Mutex::new(Vec::<String>::new()));
        let record = Arc::clone(&seen);
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = target.accept().await {
                let mut request = Vec::new();
                let mut buf = [0u8; 1024];
                while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                    match socket.read(&mut buf).await {
                        Ok(0) | Err(_) => break,
                        Ok(n) => request.extend_from_slice(&buf[..n]),
                    }
                }
                record
                    .lock()
                    .unwrap()
                    .push(String::from_utf8_lossy(&request).to_ascii_lowercase());
                let _ = socket.write_all(&ok(b"arrived")).await;
            }
        });
        let (addr, _) = serve(move |_| redirect(&format!("http://localhost:{port}/media"))).await;
        let r = internal()
            .fetch(&format!("http://{addr}/start?token=SENTINEL_SECRET"))
            .await;
        assert_eq!(r.unwrap(), b"arrived");
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert!(!seen[0].contains("referer"), "{}", seen[0]);
        assert!(!seen[0].contains("sentinel_secret"), "{}", seen[0]);
    }

    #[tokio::test]
    async fn redirect_to_a_data_url_is_a_refusal() {
        // reqwest will not follow a non-HTTP target: it returns the 302, which is
        // reported as the caller's problem, and nothing is decoded or requested.
        let (addr, connections) = serve(|_| redirect("data:image/png;base64,dGVzdA==")).await;
        let r = internal().fetch(&format!("http://{addr}/x")).await;
        assert!(matches!(r, Err(MmError::InvalidInput { .. })), "{r:?}");
        assert!(format!("{r:?}").contains("302"), "{r:?}");
        assert_eq!(connections.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn redirect_to_a_disallowed_destination_is_refused_without_a_request() {
        // The origin (127.0.0.1) is on the allowlist; it redirects to 127.0.0.2,
        // which is not. The redirect hook must refuse the hop, and the target
        // must never see a request.
        let hits = Arc::new(Mutex::new(Vec::<String>::new()));
        let seen = Arc::clone(&hits);
        let probe = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = probe.local_addr().unwrap().port();
        drop(probe);
        let origin = tokio::net::TcpListener::bind(("127.0.0.1", port))
            .await
            .unwrap();
        let victim = tokio::net::TcpListener::bind(("127.0.0.2", port))
            .await
            .unwrap();
        let to_victim = redirect(&format!("http://127.0.0.2:{port}/secret"));
        tokio::spawn(async move {
            while let Ok((mut sock, _)) = origin.accept().await {
                let mut buf = [0u8; 1024];
                let _ = sock.read(&mut buf).await;
                let _ = sock.write_all(&to_victim).await;
                let _ = sock.shutdown().await;
            }
        });
        tokio::spawn(async move {
            while let Ok((mut sock, _)) = victim.accept().await {
                let mut buf = [0u8; 1024];
                let n = sock.read(&mut buf).await.unwrap_or(0);
                seen.lock()
                    .unwrap()
                    .push(String::from_utf8_lossy(&buf[..n]).into_owned());
                let _ = sock.write_all(&ok(b"secret")).await;
            }
        });
        let guarded = fetcher(FetchPolicy {
            allowed_media_domains: Some(["127.0.0.1".to_string()].into()),
            ..FetchPolicy::with_internal_access(true)
        });
        let r = guarded
            .fetch(&format!("http://127.0.0.1:{port}/start"))
            .await;
        assert!(matches!(r, Err(MmError::InvalidInput { .. })), "{r:?}");
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(
            hits.lock().unwrap().is_empty(),
            "redirect target was contacted"
        );
        // Sanity: the same fetcher without the allowlist does follow the redirect.
        let r = internal()
            .fetch(&format!("http://127.0.0.1:{port}/start"))
            .await;
        assert_eq!(r.unwrap(), b"secret");
        assert_eq!(hits.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn transport_errors_do_not_carry_the_url() {
        let probe = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = probe.local_addr().unwrap().port();
        drop(probe);
        let err = internal()
            .fetch(&format!("http://127.0.0.1:{port}/m?token=SENTINEL_SECRET"))
            .await
            .unwrap_err();
        assert!(matches!(err, MmError::Internal { .. }), "{err:?}");
        let mut text = format!("{err} {err:?}");
        let mut cause = std::error::Error::source(&err);
        while let Some(c) = cause {
            text.push_str(&format!(" {c} {c:?}"));
            cause = c.source();
        }
        assert!(!text.contains("SENTINEL_SECRET"), "{text}");
    }

    #[tokio::test]
    async fn data_url_spellings_the_url_parser_normalises_are_refused() {
        let f = fetcher(FetchPolicy::default());
        // `Url::parse` drops tabs and newlines inside the scheme, so these look
        // like data URLs to it but not to the bounded decoder.
        for bad in [
            "da\tta:image/png;base64,AAAA",
            "d\na\rta:image/png;base64,AAAA",
        ] {
            let r = f.fetch(bad).await;
            assert!(
                matches!(r, Err(MmError::InvalidInput { .. })),
                "{bad:?}: {r:?}"
            );
        }
        // An absurdly long http URL is refused before it is parsed, by the
        // public checks too.
        let long = format!("http://example.com/{}", "a".repeat(MAX_URL_BYTES));
        let r = f.fetch(&long).await;
        assert!(matches!(r, Err(MmError::InvalidInput { .. })), "{r:?}");
        let r = f.check_url(&long);
        assert!(matches!(r, Err(MmError::InvalidInput { .. })), "{r:?}");
        let r = f.check_url_with_dns(&long).await;
        assert!(matches!(r, Err(MmError::InvalidInput { .. })), "{r:?}");
        // So are the data URL spellings `fetch` refuses.
        assert!(f.check_url("da\tta:image/png;base64,AAAA").is_err());
    }

    #[tokio::test]
    async fn a_huge_declared_content_length_neither_panics_nor_allocates_it() {
        let (addr, _) = serve(|_| {
            b"HTTP/1.1 200 OK\r\nContent-Length: 9223372036854775808\r\nConnection: close\r\n\r\n"
                .to_vec()
        })
        .await;
        let f = fetcher(FetchPolicy {
            max_bytes: u64::MAX,
            ..FetchPolicy::with_internal_access(true)
        });
        // The body never arrives; the point is no panic and no giant reservation.
        let r = f.fetch(&format!("http://{addr}/x")).await;
        assert!(r.is_err(), "{r:?}");
    }

    #[tokio::test]
    async fn reqwest_runs_the_resolver_hook_for_hostnames() {
        // Bypass `check` (which names `localhost`) and drive the policy client
        // directly: the DNS hook alone must refuse a name that resolves to
        // loopback, without ever connecting.
        let (addr, connections) = serve(|_| ok(b"secret")).await;
        let client = client_builder(&FetchPolicy::default())
            .unwrap()
            .build()
            .unwrap();
        let err = client
            .get(format!("http://localhost:{}/", addr.port()))
            .send()
            .await
            .map_err(map_reqwest)
            .unwrap_err();
        assert!(matches!(err, MmError::InvalidInput { .. }), "{err:?}");
        assert!(
            format!("{err:?}").contains("no non-blocked addresses"),
            "{err:?}"
        );
        assert_eq!(connections.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn default_policy_is_conservative() {
        let p = FetchPolicy::default();
        assert!(!p.allow_direct_ip && !p.allow_direct_port && !p.allow_private_ips);
        assert!(p.proxy.is_none());
        assert_eq!(p.timeout, Some(Duration::from_secs(30)));
        assert_eq!(p.max_bytes, MAX_FETCH_BYTES);
        assert!(p.user_agent.starts_with("dynamo-multimodal/"));
        let on_prem = FetchPolicy::with_internal_access(true);
        assert!(on_prem.allow_direct_ip && on_prem.allow_direct_port && on_prem.allow_private_ips);
        assert!(
            on_prem.proxy.is_none(),
            "internal access does not imply a proxy"
        );
    }

    #[test]
    fn a_built_fetcher_keeps_its_policy() {
        let mut policy = FetchPolicy::default();
        let f = fetcher(policy.clone());
        policy.allow_private_ips = true;
        assert!(!f.policy().allow_private_ips);
        assert!(f.check_url("http://localhost/a").is_err());
        assert!(f.check_url("not a url").is_err());
    }
}
