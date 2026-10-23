// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export function obsoleteSelection() { return new Error('Obsolete fixture selection'); }

// Shared work owns its controller. Individual consumers can withdraw without
// canceling a pack, recipe index, or source archive another consumer still needs.
export function sharedRead(cache, key, create, signal, {retain = true} = {}) {
  signal?.throwIfAborted();
  let task = cache.get(key);
  if (!task) {
    task = {controller: new AbortController(), consumers: 0, settled: false};
    task.promise = Promise.resolve().then(() => {
      task.controller.signal.throwIfAborted();
      return create(task.controller.signal);
    }).then(value => {
      task.settled = true;
      task.value = value;
      if (!retain && cache.get(key) === task) cache.delete(key);
      return value;
    }, error => {
      task.settled = true;
      if (cache.get(key) === task) cache.delete(key);
      throw error;
    });
    cache.set(key, task);
  }
  task.consumers++;
  return new Promise((resolve, reject) => {
    let released = false;
    const release = () => {
      if (released) return;
      released = true;
      signal?.removeEventListener('abort', abort);
      task.consumers--;
      if (!task.settled && task.consumers === 0) {
        if (cache.get(key) === task) cache.delete(key);
        task.controller.abort(obsoleteSelection());
      }
    };
    const abort = () => { release(); reject(signal.reason); };
    signal?.addEventListener('abort', abort, {once: true});
    task.promise.then(value => { release(); resolve(value); }, error => { release(); reject(error); });
  });
}

// The selection owns cancellable tasks; ordinary reads (including Raw) pin a
// task until it settles. Both paths still share one identity and one slot.
export class DownloadScheduler {
  constructor({limit = 8} = {}) {
    this.limit = limit;
    this.tasks = new Map();
    this.active = 0;
  }
  demand(keys) {
    const wanted = new Set(keys);
    for (const [key, task] of this.tasks) {
      if (!wanted.has(key) && !task.pinned) {
        this.tasks.delete(key);
        task.controller.abort(obsoleteSelection());
        task.reject(obsoleteSelection());
        this.finish(task);
      }
    }
    this.drain();
  }
  read(key, create, {selection = false} = {}) {
    const existing = this.tasks.get(key);
    if (existing) {
      if (!selection) existing.pinned = true;
      return existing.promise;
    }
    const task = {key, create, pinned: !selection, active: false, finished: false, controller: new AbortController()};
    task.promise = new Promise((resolve, reject) => { task.resolve = resolve; task.reject = reject; });
    this.tasks.set(key, task);
    this.drain();
    return task.promise;
  }
  finish(task) {
    if (task.finished) return;
    task.finished = true;
    if (this.tasks.get(task.key) === task) this.tasks.delete(task.key);
    if (task.active) this.active--;
  }
  drain() {
    while (this.active < this.limit) {
      const task = [...this.tasks.values()].find(item => !item.active);
      if (!task) return;
      task.active = true;
      this.active++;
      Promise.resolve().then(() => {
        task.controller.signal.throwIfAborted();
        return task.create(task.controller.signal);
      }).then(task.resolve, task.reject).finally(() => { this.finish(task); this.drain(); });
    }
  }
}
