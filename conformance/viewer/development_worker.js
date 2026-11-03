// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { installWorker } from './worker.js';
import * as yaml from './yaml_data.js';
installWorker(yaml);
