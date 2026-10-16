// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Cache population may finish after navigation, but only the latest selection paints.
export class SelectionLoader {
  constructor({ build, paint, state }) { this.build = build; this.paint = paint; this.state = state; this.generation = 0; this.pending = 0; }
  async select(tabId, selection) {
    const generation = ++this.generation;
    this.pending++;
    this.state(tabId, 'loading');
    try {
      const model = await this.build(tabId, selection);
      if (generation !== this.generation) return;
      this.paint(tabId, model);
      this.state(tabId, 'ready');
    } catch (error) {
      if (generation === this.generation) this.state(tabId, 'error', error);
    } finally { this.pending--; }
  }
}
