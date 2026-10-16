// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { readTarGzip } from './data.js';
import { parseSnapshot } from './snapshots.js';

self.addEventListener('message', async (event) => {
  const { id, operation, bytes, options } = event.data;
  try {
    let result;
    if (operation === 'snapshot') result = await parseSnapshot(bytes, options);
    else if (operation === 'archive') result = readTarGzip(bytes, options);
    else if (operation === 'yaml') result = (await import('./yaml_data.js')).parseYamlBytes(bytes, options);
    else if (operation === 'archiveCases') result = (await import('./yaml_data.js')).readArchiveCases(bytes, options);
    else throw new Error(`unknown worker operation: ${operation}`);
    self.postMessage({ id, result });
  } catch (error) {
    self.postMessage({ id, error: error instanceof Error ? error.message : String(error) });
  }
});
