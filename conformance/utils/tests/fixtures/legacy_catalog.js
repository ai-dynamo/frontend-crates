// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Synthetic legacy readers supply their own documents; the current catalog has
// no historical archive inventory to borrow for these independent unit tests.
export function syntheticLegacyInputs(catalog) {
  catalog.sources.archives = [...new Set(catalog.metadata.tabs.filter(tab => tab.suite !== 'unified').map(tab => tab.root))]
    .map(root => ({ path: `conformance/fixtures/${root}/inputs.tar.gz`, members: [], families: [], model_labels: {} }));
  catalog.sources.archives.push({ path: 'conformance/fixtures/toolcalling/fixtures-batch-on-stream-v1.tar.gz', members: [], families: [] });
  return catalog;
}
