// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { configureDevelopmentParser } from './data.js';

configureDevelopmentParser(() => import('./yaml_data.js'), new URL('./development_worker.js', import.meta.url));
