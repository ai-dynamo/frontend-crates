// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::{Map, Value};

pub(crate) fn is_blank(value: &Value) -> bool {
    value.is_null()
        || value.as_str().is_some_and(|s| s.trim().is_empty())
        || value.as_object().is_some_and(Map::is_empty)
        || value.as_array().is_some_and(Vec::is_empty)
}
