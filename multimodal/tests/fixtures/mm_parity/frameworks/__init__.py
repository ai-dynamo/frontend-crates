# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One module per framework column. Each has `run(request, args)`, which takes
the request and returns the response of README.md's worker protocol."""

import hashlib
import json
import re
import subprocess
import sys

import numpy as np


def sha256(array, dtype):
    """The digest in the table's byte order: `<i4` ids, `<f4` features,
    `<i8` positions, `|u1` pixels."""
    return hashlib.sha256(np.ascontiguousarray(array, dtype=dtype).tobytes()).hexdigest()


def versions():
    """The libraries of this environment that change a framework's values."""
    import PIL
    import torch
    import torchvision
    import transformers

    return {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "pillow": PIL.__version__,
        "numpy": np.__version__,
    }


def worker(command, request):
    """Run a worker in another environment: the request on stdin, the response on
    stdout. A message names the checkpoint `<model_dir>`, not its host path."""
    done = subprocess.run(command, input=json.dumps(request), capture_output=True, text=True)
    if done.returncode:
        sys.exit(f"{' '.join(command)}: exit {done.returncode}\n{done.stderr[-4000:]}")
    response = json.loads(done.stdout.replace(request["model_dir"], "<model_dir>"))
    assert "versions" in response, f"{command[0]}: give the versions of the worker's environment"
    return response


def reason(error):
    """A refusal as `<exception>: <message>`, without the object addresses that
    some messages carry, so that the table is byte-stable."""
    return re.sub(r" at 0x[0-9a-fA-F]+", "", f"{type(error).__name__}: {error}")


def spans(ids, token):
    """The inclusive `[start, end]` of each run of `token` in `ids`."""
    runs = []
    for i, id_ in enumerate(ids):
        if id_ != token:
            continue
        if runs and runs[-1][1] == i - 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return runs
