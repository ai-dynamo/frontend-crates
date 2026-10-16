// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import { access, readFile } from 'node:fs/promises';
import { createSourceLink } from './links.js';
const root = new URL('../../', import.meta.url);
const local = createSourceLink('http://host.test/nested/');
assert.equal(local('parsers/v1/src/tool_calling/config.rs'), 'http://host.test/nested/parsers/v1/src/tool_calling/config.rs');
const catalog = JSON.parse(await readFile(new URL('conformance/viewer-catalog.json', root)));
for (const target of Object.values(catalog.metadata.documentation)) await access(new URL(target, root));
for (const target of ['conformance/toolcalling/fixtures-v1/', 'conformance/toolcalling/fixtures-stream-v1/', 'conformance/reasoning/fixtures/']) {
  const resolved = new URL(local(target)).pathname.replace('/nested/', '');
  await access(new URL(resolved, root));
}
const publication = { schema: 1, revision: 'a'.repeat(40), catalog_revision: catalog.revision };
const web = createSourceLink('http://host.test/nested/', publication, catalog.revision);
assert.equal(web('parsers/v1/src/tool_calling/config.rs'), `https://github.com/ai-dynamo/frontend-crates/blob/${publication.revision}/parsers/v1/src/tool_calling/config.rs`);
assert.ok(web('parsers/v1/src/tool_calling/').includes(`/tree/${publication.revision}/`));
for (const revision of ['main', 'short', 'a'.repeat(39), null]) assert.throws(() => createSourceLink('http://host.test/', { ...publication, revision }, catalog.revision), /Publication/);
assert.throws(() => createSourceLink('http://host.test/', publication, 'other'), /Publication/);
for (const target of ['../outside', '/etc/passwd', 'javascript:invalid', 'a\\b', 'a//b']) assert.throws(() => local(target), /Invalid source/);
console.log('Source targets resolve under root/nested hosting and immutable publication identity');
