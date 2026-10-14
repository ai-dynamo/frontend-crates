// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Queued and active work share one identity; promotion never starts a second read.
export class DownloadScheduler {
  constructor({ limit = 8, backgroundLimit = 2 } = {}) {
    this.limit = limit; this.backgroundLimit = backgroundLimit;
    this.foregroundBusy = false; this.tasks = new Map(); this.active = 0; this.backgroundActive = 0;
  }
  demand(keys) {
    this.foregroundBusy = true;
    const wanted = new Set(keys);
    for (const [key, task] of this.tasks) {
      if (wanted.has(key)) this.promote(task);
      else if (!task.active) {
        this.tasks.delete(key); task.reject(new Error('Obsolete fixture selection'));
      }
    }
    this.drain();
  }
  ready() { this.foregroundBusy = false; this.drain(); }
  promote(task) {
    if (!task.background) return;
    task.background = false;
    if (task.active) this.backgroundActive--;
  }
  read(key, create, background = false) {
    const existing = this.tasks.get(key);
    if (existing) { if (!background) this.promote(existing); this.drain(); return existing.promise; }
    const task = { key, create, background, active: false };
    task.promise = new Promise((resolve, reject) => { task.resolve = resolve; task.reject = reject; });
    this.tasks.set(key, task); this.drain(); return task.promise;
  }
  drain() {
    while (this.active < this.limit) {
      const queued = [...this.tasks.values()].filter(task => !task.active);
      const foreground = queued.find(task => !task.background);
      const task = foreground || (!this.foregroundBusy && this.backgroundActive < this.backgroundLimit ? queued.find(task => task.background) : null);
      if (!task) return;
      task.active = true; this.active++; if (task.background) this.backgroundActive++;
      Promise.resolve().then(task.create).then(task.resolve, task.reject).finally(() => {
        this.tasks.delete(task.key); this.active--; if (task.background) this.backgroundActive--; this.drain();
      });
    }
  }
}
