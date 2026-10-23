// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { readFileSync } from 'node:fs';
import { legacyOutcome } from './comparison.js';

// Catalog indexing retains only the failure-control bit. Successful output and
// raw failure text stay in their source archives; classification has one owner.
const reasons = JSON.parse(readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(reasons.map(reason => legacyOutcome({ unavailable: reason }, []).status === 'problem')));
