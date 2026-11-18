# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic codec fixtures. Requires numpy==1.26.4, soundfile==0.13.1.

Uses soundfile's bundled libsndfile encoder; no network or model assets.
Lossless PCM is defined by integer arithmetic, independently of the decoder.
"""
import json
from pathlib import Path

import numpy as np
import soundfile as sf

assert np.__version__ == "1.26.4"
assert sf.__version__ == "0.13.1"
root = Path(__file__).parent
n = 4096
i = np.arange(n, dtype=np.int64)
pcm = np.stack([(i % 64 - 32) * 400, ((i + 16) % 64 - 32) * 400], axis=1).astype(np.int16)
for ext, fmt, subtype in [("wav", "WAV", "PCM_16"), ("flac", "FLAC", "PCM_16"),
                          ("ogg", "OGG", "VORBIS"), ("mp3", "MP3", "MPEG_LAYER_III")]:
    sf.write(root / f"stereo.{ext}", pcm, 22050, format=fmt, subtype=subtype)
    if ext in {"mp3", "ogg"}:
        decoded, rate = sf.read(root / f"stereo.{ext}", dtype="float32")
        assert decoded.shape == (n, 2) and rate == 22050
        (root / f"stereo.{ext}.decoded.f32le").write_bytes(decoded.astype("<f4").tobytes())
manifest = dict(generator=dict(numpy=np.__version__, soundfile=sf.__version__,
    libsndfile=sf.__libsndfile_version__), source_frames=n, channels=2, sample_rate=22050,
    pcm="channel 0: ((i % 64)-32)*400; channel 1: (((i+16)%64)-32)*400, int16",
    lossless=["wav", "flac"], lossy=["ogg", "mp3"],
    lossy_reference="libsndfile read(dtype=float32), interleaved little-endian f32; numerical comparison only")
(root / "codecs.json").write_text(json.dumps(manifest, indent=2) + "\n")
