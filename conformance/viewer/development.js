// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { initializeDevelopmentParser } from './yaml_data.js';
import { startApp } from './app.js';
initializeDevelopmentParser();
import { configureDevelopmentCatalog } from './app.js';
configureDevelopmentCatalog(async (baseUrl, fetchImpl) => {
  const response = await fetchImpl(new URL('conformance/viewer-unified-catalog.json', baseUrl), { cache: 'no-store' });
  if (!response.ok) throw new Error(`Catalog fetch failed (${response.status})`);
  return response.json();
});
await startApp();
