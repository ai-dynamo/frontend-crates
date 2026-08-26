// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeSet, HashMap};
use std::time::Duration;

use agent_rt_sandbox::{CommandPolicy, KubernetesSandboxConfig, KubernetesSandboxProfile};
use serde::Deserialize;
use thiserror::Error;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SandboxCatalogFile {
    pub tenant_namespaces: HashMap<String, String>,
    pub profiles: HashMap<String, SandboxProfileFile>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SandboxProfileFile {
    pub warm_pool: String,
    pub workspace_ttl_seconds: u64,
    pub max_execution_timeout_millis: u64,
    pub max_output_bytes: u64,
    pub max_artifact_bytes: u64,
    pub allowed_executables: BTreeSet<String>,
    #[serde(default)]
    pub allow_environment: bool,
    #[serde(default = "default_max_arguments")]
    pub max_arguments: usize,
    #[serde(default = "default_max_argument_bytes")]
    pub max_argument_bytes: usize,
    #[serde(default = "default_max_environment_variables")]
    pub max_environment_variables: usize,
    #[serde(default = "default_max_environment_bytes")]
    pub max_environment_bytes: usize,
    #[serde(default = "default_max_stdin_bytes")]
    pub max_stdin_bytes: usize,
    #[serde(default = "default_max_artifacts")]
    pub max_artifacts: usize,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SandboxCatalogError {
    #[error("sandbox catalog JSON is invalid: {0}")]
    Json(String),
    #[error("sandbox catalog must configure at least one tenant and profile")]
    Empty,
    #[error("sandbox catalog contains an invalid tenant ID, profile, namespace, or warm pool")]
    InvalidName,
    #[error("sandbox profile must allow at least one bounded executable")]
    InvalidExecutables,
    #[error("sandbox profile limits must be nonzero")]
    ZeroLimit,
}

impl SandboxCatalogFile {
    pub fn from_json(contents: &str) -> Result<Self, SandboxCatalogError> {
        serde_json::from_str(contents).map_err(|error| SandboxCatalogError::Json(error.to_string()))
    }
}

impl TryFrom<SandboxCatalogFile> for KubernetesSandboxConfig {
    type Error = SandboxCatalogError;

    fn try_from(file: SandboxCatalogFile) -> Result<Self, Self::Error> {
        if file.tenant_namespaces.is_empty() || file.profiles.is_empty() {
            return Err(SandboxCatalogError::Empty);
        }
        if file.tenant_namespaces.iter().any(|(tenant, namespace)| {
            !valid_scope_name(tenant) || !valid_kubernetes_name(namespace)
        }) {
            return Err(SandboxCatalogError::InvalidName);
        }
        let mut profiles = HashMap::with_capacity(file.profiles.len());
        for (name, profile) in file.profiles {
            if !valid_scope_name(&name) || !valid_kubernetes_name(&profile.warm_pool) {
                return Err(SandboxCatalogError::InvalidName);
            }
            if profile.allowed_executables.is_empty()
                || profile.allowed_executables.iter().any(|executable| {
                    executable.is_empty()
                        || executable.len() > 1_024
                        || executable.as_bytes().contains(&0)
                })
            {
                return Err(SandboxCatalogError::InvalidExecutables);
            }
            if profile.workspace_ttl_seconds == 0
                || profile.max_execution_timeout_millis == 0
                || profile.max_output_bytes == 0
                || profile.max_artifact_bytes == 0
                || profile.max_arguments == 0
                || profile.max_argument_bytes == 0
                || profile.max_environment_variables == 0
                || profile.max_environment_bytes == 0
                || profile.max_stdin_bytes == 0
                || profile.max_artifacts == 0
            {
                return Err(SandboxCatalogError::ZeroLimit);
            }
            profiles.insert(
                name,
                KubernetesSandboxProfile {
                    warm_pool: profile.warm_pool,
                    workspace_ttl: Duration::from_secs(profile.workspace_ttl_seconds),
                    max_execution_timeout: Duration::from_millis(
                        profile.max_execution_timeout_millis,
                    ),
                    max_output_bytes: profile.max_output_bytes,
                    max_artifact_bytes: profile.max_artifact_bytes,
                    command_policy: CommandPolicy {
                        allowed_executables: profile.allowed_executables,
                        allow_environment: profile.allow_environment,
                        max_arguments: profile.max_arguments,
                        max_argument_bytes: profile.max_argument_bytes,
                        max_environment_variables: profile.max_environment_variables,
                        max_environment_bytes: profile.max_environment_bytes,
                        max_stdin_bytes: profile.max_stdin_bytes,
                        max_artifacts: profile.max_artifacts,
                    },
                },
            );
        }
        Ok(Self {
            tenant_namespaces: file.tenant_namespaces,
            profiles,
        })
    }
}

fn valid_scope_name(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':' | b'@')
        })
}

fn valid_kubernetes_name(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 253
        && value
            .split('.')
            .all(|label| !label.is_empty() && label.len() <= 63 && valid_dns_label(label))
}

fn valid_dns_label(label: &str) -> bool {
    label
        .bytes()
        .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        && label
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphanumeric)
        && label
            .as_bytes()
            .last()
            .is_some_and(u8::is_ascii_alphanumeric)
}

const fn default_max_arguments() -> usize {
    64
}

const fn default_max_argument_bytes() -> usize {
    1024 * 1024
}

const fn default_max_environment_variables() -> usize {
    64
}

const fn default_max_environment_bytes() -> usize {
    256 * 1024
}

const fn default_max_stdin_bytes() -> usize {
    1024 * 1024
}

const fn default_max_artifacts() -> usize {
    32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_operator_owned_catalog() {
        let file = SandboxCatalogFile::from_json(
            r#"{
                "tenant_namespaces": {"tenant-a": "tenant-a-sandboxes"},
                "profiles": {
                    "python-deny-egress": {
                        "warm_pool": "python-deny-egress",
                        "workspace_ttl_seconds": 3600,
                        "max_execution_timeout_millis": 60000,
                        "max_output_bytes": 1048576,
                        "max_artifact_bytes": 16777216,
                        "allowed_executables": ["python"]
                    }
                }
            }"#,
        )
        .unwrap();
        let config = KubernetesSandboxConfig::try_from(file).unwrap();
        assert_eq!(config.tenant_namespaces["tenant-a"], "tenant-a-sandboxes");
        assert_eq!(
            config.profiles["python-deny-egress"].workspace_ttl,
            Duration::from_secs(3600)
        );
    }

    #[test]
    fn rejects_unbounded_or_invalid_operator_values() {
        let file = SandboxCatalogFile::from_json(
            r#"{
                "tenant_namespaces": {"tenant-a": "INVALID_NAMESPACE"},
                "profiles": {
                    "python": {
                        "warm_pool": "pool",
                        "workspace_ttl_seconds": 0,
                        "max_execution_timeout_millis": 1,
                        "max_output_bytes": 1,
                        "max_artifact_bytes": 1,
                        "allowed_executables": []
                    }
                }
            }"#,
        )
        .unwrap();
        assert!(KubernetesSandboxConfig::try_from(file).is_err());
    }
}
