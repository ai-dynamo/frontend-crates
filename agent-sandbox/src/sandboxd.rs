// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::time::Duration;

use reqwest::StatusCode;
use thiserror::Error;
use tokio::sync::watch;
use tonic::transport::{Channel, Endpoint};

use crate::{ExecutionState, SandboxClaimHandle, SandboxCommand, SandboxLimits};

mod process {
    tonic::include_proto!("process.v1");
}

use process::process_service_client::ProcessServiceClient;
use process::start_response::Event;
use process::write_stdin_request::Payload;
use process::{Empty, ProcessConfig, SendSignalRequest, Signal, StartRequest, WriteStdinRequest};

#[derive(Debug, Clone)]
pub struct SandboxdClientConfig {
    pub grpc_port: u16,
    pub rest_port: u16,
    pub connect_timeout: Duration,
    pub request_timeout: Duration,
}

impl Default for SandboxdClientConfig {
    fn default() -> Self {
        Self {
            grpc_port: 9090,
            rest_port: 8080,
            connect_timeout: Duration::from_secs(5),
            request_timeout: Duration::from_secs(15),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxdRunOutcome {
    pub state: ExecutionState,
    pub exit_code: Option<i32>,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub failure_code: Option<String>,
}

#[derive(Debug, Error)]
pub enum SandboxdClientError {
    #[error("sandboxd service FQDN is invalid")]
    InvalidServiceEndpoint,
    #[error("sandboxd gRPC endpoint is invalid: {0}")]
    InvalidGrpcEndpoint(#[from] tonic::transport::Error),
    #[error("sandboxd gRPC request failed: {0}")]
    Grpc(#[from] tonic::Status),
    #[error("sandboxd process stream ended before an init event")]
    MissingProcessId,
    #[error("sandboxd process stream ended before an exit event")]
    MissingExit,
    #[error("sandboxd file request failed: {0}")]
    FileTransport(#[from] reqwest::Error),
    #[error("sandboxd file request returned HTTP {status}: {body}")]
    FileHttp { status: u16, body: String },
    #[error("sandboxd file exceeds the configured byte limit")]
    FileTooLarge,
}

#[derive(Clone)]
pub struct SandboxdClient {
    config: SandboxdClientConfig,
    http: reqwest::Client,
}

impl SandboxdClient {
    pub fn new(config: SandboxdClientConfig) -> Result<Self, SandboxdClientError> {
        let http = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(config.request_timeout)
            .build()?;
        Ok(Self { config, http })
    }

    pub async fn execute(
        &self,
        sandbox: &SandboxClaimHandle,
        command: &SandboxCommand,
        limits: &SandboxLimits,
        mut cancellation: watch::Receiver<bool>,
    ) -> Result<SandboxdRunOutcome, SandboxdClientError> {
        let mut client = self.connect(sandbox).await?;
        let response = client
            .start(StartRequest {
                config: Some(ProcessConfig {
                    command: command.argv.clone(),
                    env_vars: command
                        .env
                        .iter()
                        .map(|(key, value)| (key.clone(), value.clone()))
                        .collect::<HashMap<_, _>>(),
                    cwd: command.cwd.clone().unwrap_or_default(),
                }),
            })
            .await?;
        let mut stream = response.into_inner();
        let process_id = match stream.message().await? {
            Some(message) => match message.event {
                Some(Event::Init(init)) => init.process_id,
                Some(Event::Stdout(_) | Event::Stderr(_) | Event::Exit(_)) | None => {
                    return Err(SandboxdClientError::MissingProcessId);
                }
            },
            None => return Err(SandboxdClientError::MissingProcessId),
        };

        if !command.stdin.is_empty()
            && client
                .write_stdin(WriteStdinRequest {
                    process_id,
                    payload: Some(Payload::Input(command.stdin.clone())),
                })
                .await
                .is_err()
        {
            return Ok(self
                .unknown_after_best_effort_kill(client, process_id)
                .await);
        }
        if client
            .write_stdin(WriteStdinRequest {
                process_id,
                payload: Some(Payload::Eof(Empty {})),
            })
            .await
            .is_err()
        {
            return Ok(self
                .unknown_after_best_effort_kill(client, process_id)
                .await);
        }

        let deadline = tokio::time::Instant::now() + Duration::from_millis(limits.timeout_millis);
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        loop {
            let message = tokio::select! {
                result = stream.message() => match result {
                    Ok(message) => message,
                    Err(_) => {
                        return Ok(self.unknown_after_best_effort_kill(client, process_id).await);
                    }
                },
                changed = cancellation.changed() => {
                    if changed.is_ok() && *cancellation.borrow() {
                        return Ok(self.terminate(client, process_id, ExecutionState::Cancelled, "cancelled").await);
                    }
                    continue;
                }
                _ = tokio::time::sleep_until(deadline) => {
                    return Ok(self.terminate(client, process_id, ExecutionState::TimedOut, "timeout").await);
                }
            };
            let Some(message) = message else {
                return Ok(self
                    .unknown_after_best_effort_kill(client, process_id)
                    .await);
            };
            match message.event {
                Some(Event::Stdout(bytes)) => {
                    if append_bounded(&mut stdout, &bytes, limits.max_output_bytes, stderr.len()) {
                        return Ok(self
                            .terminate(
                                client,
                                process_id,
                                ExecutionState::Failed,
                                "output_limit_exceeded",
                            )
                            .await);
                    }
                }
                Some(Event::Stderr(bytes)) => {
                    if append_bounded(&mut stderr, &bytes, limits.max_output_bytes, stdout.len()) {
                        return Ok(self
                            .terminate(
                                client,
                                process_id,
                                ExecutionState::Failed,
                                "output_limit_exceeded",
                            )
                            .await);
                    }
                }
                Some(Event::Exit(exit)) => {
                    let succeeded = exit.exit_code == 0;
                    return Ok(SandboxdRunOutcome {
                        state: if succeeded {
                            ExecutionState::Succeeded
                        } else {
                            ExecutionState::Failed
                        },
                        exit_code: Some(exit.exit_code),
                        stdout,
                        stderr,
                        failure_code: (!succeeded).then(|| "nonzero_exit".to_owned()),
                    });
                }
                Some(Event::Init(_)) | None => {
                    return Ok(self
                        .unknown_after_best_effort_kill(client, process_id)
                        .await);
                }
            }
        }
    }

    pub async fn read_file(
        &self,
        sandbox: &SandboxClaimHandle,
        path: &str,
        max_bytes: u64,
    ) -> Result<Vec<u8>, SandboxdClientError> {
        validate_service_fqdn(&sandbox.service_fqdn)?;
        let url = format!(
            "http://{}:{}/v1/files/{}",
            sandbox.service_fqdn,
            self.config.rest_port,
            percent_encode(path)
        );
        let response = self.http.get(url).send().await?;
        if response.status() == StatusCode::PAYLOAD_TOO_LARGE {
            return Err(SandboxdClientError::FileTooLarge);
        }
        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            return Err(SandboxdClientError::FileHttp {
                status,
                body: body.chars().take(512).collect(),
            });
        }
        let limit = usize::try_from(max_bytes).unwrap_or(usize::MAX);
        let bytes = response.bytes().await?;
        if bytes.len() > limit {
            return Err(SandboxdClientError::FileTooLarge);
        }
        Ok(bytes.to_vec())
    }

    async fn connect(
        &self,
        sandbox: &SandboxClaimHandle,
    ) -> Result<ProcessServiceClient<Channel>, SandboxdClientError> {
        validate_service_fqdn(&sandbox.service_fqdn)?;
        let endpoint = Endpoint::from_shared(format!(
            "http://{}:{}",
            sandbox.service_fqdn, self.config.grpc_port
        ))?
        .connect_timeout(self.config.connect_timeout);
        Ok(ProcessServiceClient::new(endpoint.connect().await?))
    }

    async fn terminate(
        &self,
        mut client: ProcessServiceClient<Channel>,
        process_id: i32,
        state: ExecutionState,
        failure_code: &str,
    ) -> SandboxdRunOutcome {
        let signal = client
            .send_signal(SendSignalRequest {
                process_id,
                signal: Signal::Sigkill as i32,
            })
            .await;
        if signal.is_err() {
            return unknown_outcome();
        }
        SandboxdRunOutcome {
            state,
            exit_code: None,
            stdout: Vec::new(),
            stderr: Vec::new(),
            failure_code: Some(failure_code.to_owned()),
        }
    }

    async fn unknown_after_best_effort_kill(
        &self,
        mut client: ProcessServiceClient<Channel>,
        process_id: i32,
    ) -> SandboxdRunOutcome {
        let _ = client
            .send_signal(SendSignalRequest {
                process_id,
                signal: Signal::Sigkill as i32,
            })
            .await;
        unknown_outcome()
    }
}

fn append_bounded(target: &mut Vec<u8>, chunk: &[u8], max_bytes: u64, other_len: usize) -> bool {
    let max = usize::try_from(max_bytes).unwrap_or(usize::MAX);
    let used = target.len().saturating_add(other_len);
    let remaining = max.saturating_sub(used);
    target.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
    chunk.len() > remaining
}

fn unknown_outcome() -> SandboxdRunOutcome {
    SandboxdRunOutcome {
        state: ExecutionState::OutcomeUnknown,
        exit_code: None,
        stdout: Vec::new(),
        stderr: Vec::new(),
        failure_code: Some("sandboxd_outcome_unknown".to_owned()),
    }
}

fn validate_service_fqdn(fqdn: &str) -> Result<(), SandboxdClientError> {
    if !fqdn.is_empty()
        && fqdn.len() <= 253
        && fqdn
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.'))
    {
        Ok(())
    } else {
        Err(SandboxdClientError::InvalidServiceEndpoint)
    }
}

fn percent_encode(value: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            encoded.push(char::from(byte));
        } else {
            encoded.push('%');
            encoded.push(char::from(HEX[(byte >> 4) as usize]));
            encoded.push(char::from(HEX[(byte & 0x0f) as usize]));
        }
    }
    match encoded.as_str() {
        "." => "%2E".to_owned(),
        ".." => "%2E%2E".to_owned(),
        _ => encoded,
    }
}

#[cfg(test)]
mod tests {
    use super::{append_bounded, percent_encode, validate_service_fqdn};

    #[test]
    fn output_bound_is_cumulative_across_streams() {
        let mut stdout = b"123".to_vec();
        assert!(append_bounded(&mut stdout, b"456", 5, 1));
        assert_eq!(stdout, b"1234");
    }

    #[test]
    fn file_paths_are_encoded_as_one_segment() {
        assert_eq!(percent_encode("dir/a b"), "dir%2Fa%20b");
        assert_eq!(percent_encode(".."), "%2E%2E");
    }

    #[test]
    fn service_endpoint_rejects_url_injection() {
        assert!(validate_service_fqdn("sandbox.ns.svc.cluster.local").is_ok());
        assert!(validate_service_fqdn("sandbox:9090@evil").is_err());
    }
}
