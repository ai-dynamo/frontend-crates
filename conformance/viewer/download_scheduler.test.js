// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import { DownloadScheduler } from './download_scheduler.js';
let active = 0, maxActive = 0, starts = 0;
const scheduler = new DownloadScheduler();
const releases = [];
const create = () => { starts++; active++; maxActive = Math.max(maxActive, active); return new Promise(resolve => releases.push(() => { active--; resolve('bytes'); })); };
const tasks = Array.from({ length: 12 }, (_, i) => scheduler.read(String(i), create));
await Promise.resolve(); assert.equal(starts, 8);
const shared = scheduler.read('0', create); assert.equal(shared, tasks[0]);
while (scheduler.tasks.size) { releases.splice(0).forEach(release => release()); await new Promise(resolve => setImmediate(resolve)); }
await Promise.all(tasks); assert.equal(maxActive, 8); assert.equal(starts, 12);
const retry = new DownloadScheduler();
await assert.rejects(retry.read('fail', async () => { throw new Error('transport'); }), /transport/);
await new Promise(resolve => setImmediate(resolve));
assert.equal(await retry.read('fail', async () => 'recovered'), 'recovered');

const stale = new DownloadScheduler({limit: 2});
let obsoleteStarted = false;
const old = stale.read('old', signal => new Promise((resolve, reject) => {
  signal.addEventListener('abort', () => reject(signal.reason), {once: true});
}), {selection: true});
const raw = stale.read('raw', () => new Promise(resolve => { globalThis.releaseRaw = resolve; }));
const obsolete = stale.read('queued', async () => { obsoleteStarted = true; }, {selection: true});
await Promise.resolve();
stale.demand(['new']);
const rejected = Promise.all([assert.rejects(old, /Obsolete/), assert.rejects(obsolete, /Obsolete/)]);
assert.equal(await stale.read('new', async () => 'current', {selection: true}), 'current');
await rejected;
assert.equal(obsoleteStarted, false);
globalThis.releaseRaw('raw');
assert.equal(await raw, 'raw');
console.log('Bounded scheduler, obsolete cancellation, pinned Raw, and retry contracts passed');
