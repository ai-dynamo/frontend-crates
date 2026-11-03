// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { readTarGzip } from './data.js';
import { parseSnapshot } from './data.js';

export function installWorker(yaml = null) {
self.addEventListener('message', async (event) => {
  const { id, operation, bytes, options } = event.data;
  try {
    const metrics = { decoding: 0, validation: 0 };
    let result;
    if (operation === 'snapshot') result = await parseSnapshot(bytes, options, metrics);
    else if (operation === 'archive') result = readTarGzip(bytes, options);
    else if (operation === 'yaml') result = yaml.parseYamlBytes(bytes, options);
    else if (operation === 'archiveCases') result = yaml.readArchiveCases(bytes, options);
    else throw new Error(`unknown worker operation: ${operation}`);
    self.postMessage({ id, result, metrics });
  } catch (error) {
    self.postMessage({ id, error: error instanceof Error ? error.message : String(error) });
  }
});

}
if (!new URL(self.location.href).pathname.endsWith('/development_worker.js')) installWorker();
