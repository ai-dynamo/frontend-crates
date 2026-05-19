<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Security Policy

## Reporting a Vulnerability

NVIDIA takes the security of our software products seriously. If you believe you have found a security vulnerability in any of the crates in this repository (`dynamo-protocols`, `dynamo-tokenizers`, `dynamo-parsers`, or the bundled demo server), please report it through coordinated disclosure to the NVIDIA Product Security Incident Response Team (PSIRT).

**Please do not report security vulnerabilities through public GitHub issues, discussions, or pull requests.**

Instead, please report them by either:

- Email: **psirt@nvidia.com**
- Web form: [NVIDIA Security Vulnerability Submission Form](https://www.nvidia.com/en-us/security/psirt-vulnerability-submission/)

For sensitive disclosures, NVIDIA's PGP key is available on the same page.

When reporting, please include as much of the following as you can:

- The affected crate(s) and version(s) (e.g. `dynamo-protocols 0.1.0`).
- A description of the vulnerability and the impact.
- Step-by-step reproduction instructions, or a minimal proof-of-concept.
- Any known mitigations or workarounds.

Once a report is received, the PSIRT team will coordinate with the maintainers of this repository on triage, fix development, and coordinated disclosure timing.

## Supported Versions

This repository is in active development; we generally support and accept security fixes against the latest published version of each crate on crates.io. Older versions are not supported.

| Crate                | Supported     |
|----------------------|---------------|
| `dynamo-protocols`   | latest only   |
| `dynamo-tokenizers`  | latest only   |
| `dynamo-parsers`     | latest only   |

## More Information

- NVIDIA Product Security: <https://www.nvidia.com/en-us/security/>
- NVIDIA PSIRT Policy: <https://www.nvidia.com/en-us/security/psirt-policies/>
