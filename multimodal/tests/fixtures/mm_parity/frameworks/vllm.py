# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""vLLM's Python multimodal processor, out of process: its image decode,
`MultiModalProcessor.apply` on a token prompt, and its model's M-RoPE."""

import base64
import hashlib
import importlib.metadata
import json
import os
import re
import sys
from pathlib import Path

SOURCE = (
    "vLLM MediaConnector.fetch_image (ImageMediaIO) on base64 data URLs, then "
    "MULTIMODAL_REGISTRY.create_processor(ModelConfig(dtype=float32, "
    "mm_device_do_normalize=False)).apply on the token prompt with per-request "
    "mm_processor_kwargs (pil backend: use_fast=False), then the model class's "
    "get_mrope_input_positions"
)


def run(request, args):
    """The driver side: this file runs as the worker under vLLM's interpreter."""
    from frameworks import worker

    return worker([args.vllm_python, "-I", __file__], request)


def sha256(array, dtype):
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(array, dtype=dtype).tobytes()).hexdigest()


def versions():
    """`frameworks.versions()`, for this environment."""
    import numpy
    import PIL
    import torch
    import torchvision
    import transformers

    return {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "pillow": PIL.__version__,
        "numpy": numpy.__version__,
    }


class VllmPath:
    """The renderer's multimodal path for one checkpoint, on the CPU."""

    def __init__(self, model_dir, max_model_len=None):
        from vllm.config import ModelConfig
        from vllm.model_executor.model_loader import get_model_architecture
        from vllm.multimodal import MULTIMODAL_REGISTRY
        from vllm.multimodal.media import MediaConnector

        # By default both families send uint8 patches and normalize in the ViT
        # (x * w + b), and `auto` casts pixel_values to bf16: keep HF's float32.
        self.config = ModelConfig(
            model=model_dir,
            dtype="float32",
            mm_device_do_normalize=False,
            # A checkpoint may ship its config and processor as remote code (Kimi-K3).
            trust_remote_code=True,
            **({"max_model_len": max_model_len} if max_model_len else {}),
        )
        self.processor = MULTIMODAL_REGISTRY.create_processor(self.config)
        self.connector = MediaConnector()
        model_cls, _ = get_model_architecture(self.config)
        self.rope = None
        if self.config.uses_mrope:
            # get_mrope_input_positions reads only `config`: no modules, no weights.
            self.rope = object.__new__(model_cls)
            self.rope.config = self.config.hf_config
        else:
            print(
                f"{model_cls.__name__}: ModelConfig.uses_mrope is False (no mrope_section "
                "in the HF config), so the model runner's get_rope_state returns None and "
                "the model gets 1-D positions",
                file=sys.stderr,
            )

    def process(self, case, images_dir):
        """The framework's own steps: decode, then the processor."""
        from vllm.multimodal.processing import ProcessorInputs, TimingContext
        from vllm.utils.torch_utils import set_default_torch_num_threads

        images = [
            self.connector.fetch_image(
                "data:image/*;base64," + base64.b64encode((images_dir / name).read_bytes()).decode()
            )
            for name in case["images"]
        ]
        kwargs = dict(case["image_kwargs"])
        input_ids = case["input_ids"]
        hf_config = self.config.hf_config
        if hasattr(hf_config, "image_placeholder"):
            # The prompt update targets the text of the placeholder (Kimi-K3), not its id.
            tokenizer = self.processor.info.get_tokenizer()
            text = tokenizer.encode(hf_config.image_placeholder, add_special_tokens=False)
            pad = hf_config.media_placeholder_token_id
            input_ids = [i for id_ in input_ids for i in (text if id_ == pad else [id_])]
        if case["backend"] == "pil":
            # `backend="pil"` also reaches the video processor, which has no pil class.
            kwargs["use_fast"] = False
        items = self.processor.info.parse_mm_data({"image": images})
        inputs = ProcessorInputs(input_ids, items, hf_processor_mm_kwargs=kwargs)
        with set_default_torch_num_threads():
            return self.processor.apply(inputs, TimingContext(enabled=False))

    def record(self, out):
        from vllm.multimodal.inputs import MultiModalBatchedField, MultiModalFeatureSpec

        input_ids = list(out["prompt_token_ids"])
        items = out["mm_kwargs"].get("image", [])
        placeholders = out["mm_placeholders"].get("image", [])
        features = []
        for item in items:
            pixels = item["pixel_values" if "pixel_values" in item else "pixel_values_flat"]
            # A batched field's item is one element of the batch: keep its batch axis.
            batched = isinstance(pixels.field, MultiModalBatchedField)
            features.append(pixels.data[None].numpy() if batched else pixels.data.numpy())
        item_offsets = [
            [list(span) for span in placeholder.extract_embeds_range()]
            for placeholder in placeholders
        ]
        output = {
            "input_ids": input_ids,
            "item_offsets": item_offsets,
            "feature_shapes": [list(feature.shape) for feature in features],
            "feature_dtypes": [str(feature.dtype) for feature in features],
            "feature_sha256": [sha256(feature, "<f4") for feature in features],
            "mrope_sha256": None,
            "mrope_delta": None,
        }
        if all(len(spans) == 1 for spans in item_offsets):
            output["offsets"] = [spans[0] for spans in item_offsets]
        grids = [item.get("image_grid_thw", item.get("grid_thws")) for item in items]
        if None not in grids:
            output["image_grid_thw"] = [grid.data.tolist() for grid in grids]
        if items and all("image_sizes" in item for item in items):
            output["image_sizes"] = [item["image_sizes"].data.flatten().tolist() for item in items]
        if self.rope is not None:
            specs = [
                MultiModalFeatureSpec(
                    data=item, modality="image", identifier=mm_hash, mm_position=placeholder
                )
                for item, mm_hash, placeholder in zip(
                    items, out["mm_hashes"]["image"], placeholders, strict=True
                )
            ]
            positions, delta = self.rope.get_mrope_input_positions(input_ids, specs)
            output["mrope_sha256"] = sha256(positions.numpy(), "<i8")
            output["mrope_delta"] = int(delta)
        return output


def respond(request):
    response = {"version": None, "source": SOURCE, "omits": []}
    try:
        response["version"] = importlib.metadata.version("vllm")
        # One processor per engine context length (`--max-model-len`) the cases name.
        paths = {}
        for case in request["cases"]:
            length = case.get("max_model_len")
            if length not in paths:
                paths[length] = VllmPath(request["model_dir"], length)
    except Exception as e:
        reason = f"vLLM cannot run: {type(e).__name__}: {e}"
        response["cases"] = {case["name"]: {"n/a": reason} for case in request["cases"]}
        return response

    images_dir = Path(request["images_dir"])
    response["cases"] = {}
    for case in request["cases"]:
        path = paths[case.get("max_model_len")]
        try:
            out = path.process(case, images_dir)
        except Exception as e:
            # Some messages carry object addresses; the response must be byte-stable.
            reason = re.sub(r" at 0x[0-9a-fA-F]+", "", f"{type(e).__name__}: {e}")
            response["cases"][case["name"]] = {"rejects": reason}
            continue
        response["cases"][case["name"]] = {"output": path.record(out)}
    return response


def main():
    request = json.load(sys.stdin)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # vLLM logs to stdout by default: keep the real stdout for the response alone.
    stdout = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    json.dump({**respond(request), "versions": versions()}, stdout)
    stdout.write("\n")
    stdout.close()


if __name__ == "__main__":
    main()
