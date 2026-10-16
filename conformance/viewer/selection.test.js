// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import { SelectionLoader } from './selection.js';
const pending = [], paints = [], states = [];
const loader = new SelectionLoader({ build: (tab, selection) => new Promise((resolve, reject) => pending.push({ tab, selection, resolve, reject })), paint: (tab, model) => paints.push([tab, model]), state: (...args) => states.push(args) });
const old = loader.select('tab-a', { reference: 'old', selected: ['old'] });
const newer = loader.select('tab-a', { reference: 'new', selected: ['new', 'peer'] });
pending[1].resolve('new-model'); await newer;
pending[0].resolve('old-model'); await old;
assert.deepEqual(paints, [['tab-a', 'new-model']]);
const abandoned = loader.select('tab-a', { selected: ['slow'] });
const active = loader.select('tab-b', { selected: [] });
pending[3].resolve('empty-reference-model'); await active;
pending[2].reject(new Error('old load failed')); await abandoned;
assert.equal(states.filter(([, state]) => state === 'error').length, 0);
assert.deepEqual(paints.at(-1), ['tab-b', 'empty-reference-model']);
const failed = loader.select('tab-b', { selected: ['missing'] });
pending[4].reject(new Error('transport')); await failed;
assert.equal(states.at(-1)[1], 'error');
const retry = loader.select('tab-b', { selected: ['missing'] });
pending[5].resolve('recovered'); await retry;
assert.deepEqual(paints.at(-1), ['tab-b', 'recovered']);
console.log('Selection generations reject stale paints and failures; empty Reference and Retry recover');
