// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The boolean spellings that Dynamo accepts for a thinking alias.
//!
//! This is a copy of `is_truthy` and `parse_bool_opt` from Dynamo's
//! `dynamo-truthy` crate (`lib/truthy/src/lib.rs`). That crate is not
//! published, and this crate sits below Dynamo, so it cannot depend on it.
//! The thinking rules must read `"on"` and `"yes"` exactly as the Dynamo
//! frontend does, so keep the two lists the same as Dynamo's.

/// `1 | true | on | yes`, case-insensitive, surrounding whitespace ignored.
pub(crate) fn is_truthy(val: &str) -> bool {
    matches!(
        val.trim().to_lowercase().as_str(),
        "1" | "true" | "on" | "yes"
    )
}

/// `0 | false | off | no` or empty, case-insensitive, surrounding whitespace
/// ignored.
fn is_falsey(val: &str) -> bool {
    matches!(
        val.trim().to_lowercase().as_str(),
        "" | "0" | "false" | "off" | "no"
    )
}

/// `Some(true)` for a truthy value, `Some(false)` for `0 | false | off | no`,
/// and `None` for an empty or unknown value.
pub(crate) fn parse_bool_opt(val: &str) -> Option<bool> {
    if is_truthy(val) {
        Some(true)
    } else if val.trim().is_empty() {
        None
    } else if is_falsey(val) {
        Some(false)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mirrors the tables in Dynamo's `lib/truthy/src/lib.rs` tests.
    const TRUTHY: &[&str] = &[
        "1", "true", "TRUE", "True", "on", "ON", "On", "yes", "YES", "Yes", " true ", "\tyes\n",
    ];
    const FALSEY: &[&str] = &[
        "0", "false", "FALSE", "False", "off", "OFF", "Off", "no", "NO", "No", "", "  ", " false ",
    ];
    const NEITHER: &[&str] = &[
        "2", "enabled", "disabled", "maybe", "y", "n", "t", "f", "-1",
    ];

    #[test]
    fn truthy_spellings() {
        for val in TRUTHY {
            assert!(is_truthy(val), "value={val:?}");
            assert!(!is_falsey(val), "value={val:?}");
        }
    }

    #[test]
    fn falsey_spellings() {
        for val in FALSEY {
            assert!(is_falsey(val), "value={val:?}");
            assert!(!is_truthy(val), "value={val:?}");
        }
    }

    #[test]
    fn invalid_spellings() {
        for val in NEITHER {
            assert!(!is_truthy(val), "value={val:?}");
            assert!(!is_falsey(val), "value={val:?}");
        }
    }

    #[test]
    fn parse_bool_opt_spellings() {
        for val in TRUTHY {
            assert_eq!(parse_bool_opt(val), Some(true), "value={val:?}");
        }
        for val in FALSEY {
            let expected = if val.trim().is_empty() {
                None // declared-but-empty is not a deliberate choice
            } else {
                Some(false)
            };
            assert_eq!(parse_bool_opt(val), expected, "value={val:?}");
        }
        for val in NEITHER {
            assert_eq!(parse_bool_opt(val), None, "value={val:?}");
        }
    }
}
