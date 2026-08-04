# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Route PyYAML through libyaml's C parser/emitter. Import for the side effect.

The conformance render is dominated by YAML: the resolvers parse the whole fixture
corpus and re-emit every staged file, and the generator re-loads it per peer version.
PyYAML's pure-Python SafeLoader/SafeDumper are what that time is actually spent in —
in one resolver run the emitter alone was 72% of tottime. CSafeLoader/CSafeDumper
produce identical documents (verified byte-identical across the staged corpus), so
this patches `yaml.safe_load` / `yaml.safe_dump` at module level ONCE.

Patching the module rather than each call site is deliberate: fixtures.py, markers.py,
and the tables package call `yaml.safe_load` directly at call time, so one import here
covers them without a copy of the shim in each file. Every module that reads or writes
fixture YAML on the render path imports this instead of rolling its own.
"""
import yaml

if hasattr(yaml, "CSafeLoader"):

    def _fast_load(stream, _loader=yaml.CSafeLoader):
        return yaml.load(stream, Loader=_loader)

    yaml.safe_load = _fast_load

if hasattr(yaml, "CSafeDumper"):

    def _fast_dump(data, stream=None, _dumper=yaml.CSafeDumper, **kwargs):
        return yaml.dump(data, stream, Dumper=_dumper, **kwargs)

    yaml.safe_dump = _fast_dump
