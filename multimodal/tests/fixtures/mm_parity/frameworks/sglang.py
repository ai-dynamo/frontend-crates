# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from sgl-project/sglang `rust/sglang-processor/tests/scripts/generate_mm_parity.py`
# (Apache-2.0).
"""SGLang's Python CPU image path: `process_mm_data_async` over the image bytes."""

import asyncio
import subprocess
from pathlib import Path
from types import SimpleNamespace

import transformers

from frameworks import reason, sha256

SOURCE = "process_mm_data_async(image bytes, token ids) on the CPU, PIL decode"


class Unloadable(Exception):
    """SGLang's config loader refuses the checkpoint."""


class SglangPath:
    """SGLang's processor over one checkpoint, one image backend and one set of
    `--mm-process-config` image overrides, on the CPU."""

    def __init__(self, model_dir, backend, overrides):
        from sglang.test.test_utils import maybe_stub_sgl_kernel

        maybe_stub_sgl_kernel()
        from sglang.srt.hf_transformers.config import get_config
        from sglang.srt.hf_transformers.processor import get_processor
        from sglang.srt.managers.multimodal_processor import (
            get_mm_processor_cls,
            import_processors,
        )
        from sglang.srt.runtime_context import publish
        from sglang.srt.server_args import ServerArgs

        try:
            get_config(str(model_dir), trust_remote_code=False)
        except (OSError, ValueError) as error:
            raise Unloadable(str(error).splitlines()[0]) from error
        mm_process_config = {"image": overrides} if overrides else {}
        server_args = SimpleNamespace(
            model_impl="sglang",
            mm_feature_transport="cpu",
            mm_enable_dp_encoder=False,
            image_processor_backend=backend,
            disable_fast_image_processor=False,
            skip_tokenizer_init=False,
            mm_preprocess_cache_size_mb=0,
            trust_mm_content_hashes=False,
            tp_size=1,
            dist_init_addr=None,
            mm_process_config=mm_process_config,
            mm_io_worker_num=1,
            mm_processor_worker_num=1,
            tokenizer_worker_num=1,
            base_gpu_id=0,
            rl_on_policy_target=None,
            allowed_media_domains=[],
            media_url_max_file_size_mb=64,
        )
        publish(
            ServerArgs(
                model_path=str(model_dir),
                model_impl="sglang",
                mm_feature_transport="cpu",
                mm_process_config=mm_process_config,
                image_processor_backend=backend,
            ),
            role="tokenizer",
        )
        hf_config = transformers.AutoConfig.from_pretrained(model_dir)
        processor = get_processor(str(model_dir), image_processor_backend=backend)
        import_processors("sglang.srt.multimodal.processors")
        processor_cls = get_mm_processor_cls(hf_config)
        self.processor = processor_cls(hf_config, server_args, processor, None, skip_mm_pool=True)
        # The CPU path: PIL decode (read by a classmethod), and CPU image kernels.
        processor_cls.gpu_image_decode = False
        self.processor._fast_image_processor_device = lambda _processor: "cpu"

    def close(self):
        from sglang.srt.runtime_context import reset_context

        self.processor.io_executor.shutdown()
        self.processor.cpu_executor.shutdown()
        reset_context()

    def process(self, input_ids, images):
        """The framework's own steps: decode, then the processor."""
        return asyncio.run(
            self.processor.process_mm_data_async(
                image_data=images,
                input_text=input_ids,
                request_obj=SimpleNamespace(
                    video_data=None, audio_data=None, rid="parity", mm_content_hashes=None
                ),
            )
        )

    def record(self, output):
        features = [item.feature.detach().cpu().float().numpy() for item in output.mm_items]
        mrope = output.mrope_positions
        # The inclusive spans of each item's feature tokens.
        item_offsets = [[list(span) for span in item.offsets] for item in output.mm_items]
        out = {
            "input_ids": [int(id_) for id_ in output.input_ids],
            "item_offsets": item_offsets,
            "image_grid_thw": [self.grid(item) for item in output.mm_items],
            "feature_shapes": [list(feature.shape) for feature in features],
            "feature_sha256": [sha256(feature, "<f4") for feature in features],
            "mrope_sha256": None if mrope is None else sha256(mrope.cpu().numpy(), "<i8"),
            "mrope_delta": None if mrope is None else int(output.mrope_position_delta.item()),
        }
        if all(len(spans) == 1 for spans in item_offsets):
            out["offsets"] = [spans[0] for spans in item_offsets]
        return out

    @staticmethod
    def grid(item):
        return item.image_grid_thw.flatten().tolist()


def version():
    import sglang

    checkout = Path(sglang.__file__).parent
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "--short=10", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return f"{sglang.__version__} ({commit})" if commit else sglang.__version__


def run(request, args, path=SglangPath):
    """`path` is `SglangPath` or a family's subclass of it."""
    images_dir = Path(request["images_dir"])
    paths, cases = {}, {}
    for case in request["cases"]:
        overrides = case["image_kwargs"] or None
        key = (case["backend"], repr(overrides))
        if key not in paths:
            try:
                paths[key] = path(request["model_dir"], case["backend"], overrides)
            except Unloadable as error:
                paths[key] = {"n/a": f"SGLang cannot load the checkpoint: {error}"}
        if isinstance(paths[key], dict):
            cases[case["name"]] = paths[key]
            continue
        images = [(images_dir / name).read_bytes() for name in case["images"]]
        try:
            output = paths[key].process(case["input_ids"], images)
        except Exception as error:  # noqa: BLE001 -- the refusal is the recorded result
            cases[case["name"]] = {"rejects": reason(error)}
            continue
        cases[case["name"]] = {"output": paths[key].record(output)}
    for opened in paths.values():
        if not isinstance(opened, dict):
            opened.close()
    return {"version": version(), "source": SOURCE, "omits": [], "cases": cases}
