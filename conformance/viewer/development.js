// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import './development_parser.js';
import { configureDevelopmentCatalog } from './startup.js';
configureDevelopmentCatalog(async (baseUrl, fetchImpl) => {
  const response = await fetchImpl(new URL('conformance/viewer-unified-catalog.json', baseUrl), { cache: 'no-store' });
  if (!response.ok) throw new Error(`Catalog fetch failed (${response.status})`);
  return response.json();
});
await import('./app.js');
