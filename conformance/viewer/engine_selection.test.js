// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import { EngineSelection } from './engine_selection.js';
import { DownloadScheduler } from './download_scheduler.js';
const descriptors = [['dynamo', 'dynamo_v2', '2'], ['dynamo-old', 'dynamo_v2', '1'], ['sglang', 'sglang_python', '2'], ['vllm', 'vllm_python', '2'], ['rust', 'vllm_rust', '2'], ['golden', 'golden', null]].map(([key, implementation, version]) => ({ source: { implementation, version, id: `${implementation}@unified@${version}` }, control: { key, impl: implementation.startsWith('dynamo') ? 'dynamo' : implementation.startsWith('sglang') ? 'sglang' : implementation === 'golden' ? 'golden' : 'vllm' } }));
const defaults = new EngineSelection(descriptors, new URL('http://localhost/?keep=1#anchor'));
assert.deepEqual(defaults.selection(), { reference: 'dynamo', selected: ['dynamo', 'sglang', 'vllm'] });
defaults.change('dynamo', '');
assert.deepEqual(defaults.selection(), { reference: null, selected: ['sglang', 'vllm'] });
const saved = defaults.serialize(new URL('http://localhost/?keep=1#anchor'));
assert.equal(saved.hash, '#anchor'); assert.equal(saved.searchParams.get('keep'), '1');
assert.deepEqual(new EngineSelection(descriptors, saved).selection(), defaults.selection());
for (const [query, expected] of [
 ['cmp=vllm', { reference: null, selected: ['vllm'] }],
 ['ref=rust&cmp=vllm,dynamo-old', { reference: 'rust', selected: ['dynamo-old', 'rust'] }],
 ['ref=golden&cmp=sglang', { reference: 'golden', selected: ['sglang'] }],
 ['ref=bad&cmp=bad,vllm', { reference: 'dynamo', selected: ['dynamo', 'vllm'] }],
 ['base_tab-unified=&cmp_tab-unified=', { reference: null, selected: [] }],
]) assert.deepEqual(new EngineSelection(descriptors, new URL(`http://localhost/?${query}`)).selection(), expected);
let active = 0, maxActive = 0, starts = 0;
const scheduler = new DownloadScheduler();
const releases = [];
const create = () => { starts++; active++; maxActive = Math.max(maxActive, active); return new Promise(resolve => releases.push(() => { active--; resolve('bytes'); })); };
const tasks = Array.from({ length: 12 }, (_, i) => scheduler.read(String(i), create));
await Promise.resolve(); assert.equal(starts, 8);
const shared = scheduler.read('0', create); assert.equal(shared, tasks[0]);
while (scheduler.tasks.size) { releases.splice(0).forEach(release => release()); await new Promise(resolve => setImmediate(resolve)); }
await Promise.all(tasks); assert.equal(maxActive, 8); assert.equal(starts, 12);
const background = new DownloadScheduler();
const bgReleases = [];
const bgCreate = () => new Promise(resolve => bgReleases.push(resolve));
const bg = [0, 1, 2].map(i => background.read(String(i), bgCreate, true));
await Promise.resolve(); assert.equal(background.backgroundActive, 2);
background.demand(['2']);
assert.equal(background.read('2', bgCreate), bg[2]);
await Promise.resolve(); assert.equal(background.active, 3); assert.equal(background.backgroundActive, 2);
bgReleases.forEach(resolve => resolve()); await Promise.all(bg);
const retry = new DownloadScheduler();
await assert.rejects(retry.read('fail', async () => { throw new Error('transport'); }, true), /transport/);
await new Promise(resolve => setImmediate(resolve));
assert.equal(await retry.read('fail', async () => 'recovered'), 'recovered');
console.log('Engine selection and bounded scheduler contracts passed');

const stale = new DownloadScheduler({ limit: 3, backgroundLimit: 2 });
const oldRelease = []; let obsoleteStarted = false;
const old = ['old-a', 'old-b'].map(key => stale.read(key, () => new Promise(resolve => oldRelease.push(resolve)), true));
const obsolete = stale.read('old-c', async () => { obsoleteStarted = true; }, true);
stale.demand(['new']);
const newSelection = stale.read('new', async () => 'current');
await assert.rejects(obsolete, /Obsolete/);
assert.equal(await newSelection, 'current');
stale.ready(); oldRelease.forEach(resolve => resolve('old')); await Promise.all(old);
await new Promise(resolve => setImmediate(resolve));
assert.equal(obsoleteStarted, false);
