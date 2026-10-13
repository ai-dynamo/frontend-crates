// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { parseYamlBytes, readArchiveCases, readTarGzip } from './data.js';

self.addEventListener('message', (event) => {
  const { id, operation, bytes, options } = event.data;
  try {
    let result;
    if (operation === 'archive') result = readTarGzip(bytes, options);
    else if (operation === 'yaml') result = parseYamlBytes(bytes, options);
    else if (operation === 'archiveCases') result = readArchiveCases(bytes, options);
    else throw new Error(`unknown worker operation: ${operation}`);
    self.postMessage({ id, result });
  } catch (error) {
    self.postMessage({ id, error: error instanceof Error ? error.message : String(error) });
  }
});
