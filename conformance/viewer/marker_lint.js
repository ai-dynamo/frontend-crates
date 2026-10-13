// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { colorizeMarkup } = require('../utils/src/assets/colorize.js');
const { family, markers } = JSON.parse(readFileSync(0, 'utf8'));
const declared = markers[family] || {};
const failures = [];
for (const [open, close] of declared.pairs || []) {
  if (colorizeMarkup(`${open}x${close}`, family, markers).includes('tt-orphan')) {
    failures.push(`pair ${JSON.stringify(open)}/${JSON.stringify(close)} renders as tt-orphan`);
  }
}
for (const token of declared.singletons || []) {
  if (colorizeMarkup(String(token), family, markers).includes('tt-orphan')) {
    failures.push(`singleton ${JSON.stringify(token)} renders as tt-orphan`);
  }
}
process.stdout.write(JSON.stringify(failures));
