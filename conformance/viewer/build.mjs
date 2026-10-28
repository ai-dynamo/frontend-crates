// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

if (process.argv.includes('--help')) {
  process.stdout.write('Bundle production app and worker to JSON on stdout; CONFORMANCE_ESBUILD_ROOT selects the locked npm installation.\n');
  process.exit(0);
}
const directory = dirname(fileURLToPath(import.meta.url));
const root = resolve(directory, '../..');
const manifest = JSON.parse(readFileSync(resolve(directory, 'package.json'), 'utf8'));
const installation = process.env.CONFORMANCE_ESBUILD_ROOT || directory;
const esbuild = await import(pathToFileURL(resolve(installation, 'node_modules/esbuild/lib/main.js')));
if (esbuild.version !== manifest.devDependencies.esbuild) throw new Error('Production bundler version differs from the lockfile owner');
const license = readFileSync(resolve(directory, 'vendor/fflate/LICENSE'), 'utf8');
const files = {};
const inputs = new Set();
let workerSource;
for (const entry of ['worker', 'app']) {
  const result = await esbuild.build({
    absWorkingDir: root,
    entryPoints: [`conformance/viewer/${entry}.js`],
    outfile: `${entry}.js`,
    bundle: true, write: false, metafile: true, minify: true,
    // Node-only resource readers share data.js and are removed by tree shaking.
    format: entry === 'worker' ? 'iife' : 'esm', platform: 'browser', target: 'es2022', external: ['node:*'],
    define: entry === 'app' ? { __CONFORMANCE_WORKER_SOURCE__: JSON.stringify(workerSource) } : {},
    legalComments: 'eof',
    banner: { js: '// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n// SPDX-License-Identifier: Apache-2.0\n/*!\n' + license + '*/' },
  });
  if (result.outputFiles.length !== 1) throw new Error('Unexpected production bundle outputs');
  if (entry === 'worker') workerSource = result.outputFiles[0].text;
  files[`conformance/viewer/${entry}.js`] = Buffer.from(result.outputFiles[0].contents).toString('base64');
  for (const input of Object.keys(result.metafile.inputs)) inputs.add(input);
}
process.stdout.write(JSON.stringify({ schema: 1, files, inputs: [...inputs].sort() }));
