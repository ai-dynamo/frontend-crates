# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The baseline: the checkpoint's own transformers processor (`AutoProcessor`)
on the CPU, fed by transformers' own image loader."""

import io

import torch
import transformers
from PIL import Image
from transformers.image_utils import load_image

from frameworks import reason, sha256, spans

SOURCE = (
    "AutoProcessor(text, images=load_image(bytes)), and Qwen2-VL's get_rope_index "
    "when config.json has an mrope_section"
)


def processor(model_dir, backend):
    """The default processor, with the image processor of `backend` when the
    default backend is another one."""
    processor = transformers.AutoProcessor.from_pretrained(model_dir)
    if processor.image_processor.backend != backend:
        processor.image_processor = transformers.AutoImageProcessor.from_pretrained(
            model_dir, backend=backend
        )
    return processor


def positions(config, ids, grids, image_token):
    """Qwen2-VL's M-RoPE positions, or None for 1-D positions."""
    if config is None:
        return None
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel

    assert config.model_type == "qwen2_vl", f"{config.model_type}: hf.py has only Qwen2-VL's M-RoPE"

    with torch.device("meta"):
        model = Qwen2VLModel(config)
    ids = torch.tensor([ids])
    positions, delta = model.get_rope_index(
        ids,
        (ids == image_token).int(),
        image_grid_thw=torch.tensor(grids),
        attention_mask=torch.ones_like(ids),
    )
    return positions[:, 0, :], int(delta.reshape(-1)[0])


def process(processor, case, text, data):
    """The framework's own steps: decode, then the processor."""
    images = [load_image(Image.open(io.BytesIO(image))) for image in data]
    return processor(
        text=[text], images=images, images_kwargs=case["image_kwargs"], return_tensors="pt"
    )


def record(processor, config, out):
    ids = out["input_ids"][0].tolist()
    pixel_values = out["pixel_values"].float().numpy()
    if "image_sizes" in out:
        return padded_batch(processor, ids, pixel_values, out["image_sizes"].tolist())
    grids = out["image_grid_thw"].tolist()
    features, start = [], 0
    for t, h, w in grids:
        features.append(pixel_values[start : start + t * h * w])
        start += t * h * w
    assert start == len(pixel_values)
    mrope = positions(config, ids, grids, processor.image_token_id)
    return {
        "input_ids": ids,
        "offsets": spans(ids, processor.image_token_id),
        "image_grid_thw": grids,
        "feature_shapes": [list(feature.shape) for feature in features],
        "feature_sha256": [sha256(feature, "<f4") for feature in features],
        "mrope_sha256": None if mrope is None else sha256(mrope[0], "<i8"),
        "mrope_delta": None if mrope is None else mrope[1],
    }


def padded_batch(processor, ids, pixel_values, sizes):
    """A batch padded to `[N, 3, H, W]` (Pixtral): each image is the top-left
    crop of its size, and its rows of image tokens end at the image end token."""
    features = [pixel_values[i : i + 1, :, :h, :w] for i, (h, w) in enumerate(sizes)]
    items = [[]]
    for span in spans(ids, processor.image_token_id):
        items[-1].append(span)
        if ids[span[1] + 1] == processor.image_end_token_id:
            items.append([])
    return {
        "input_ids": ids,
        "item_offsets": items[:-1],
        "image_sizes": sizes,
        "feature_shapes": [list(feature.shape) for feature in features],
        "feature_sha256": [sha256(feature, "<f4") for feature in features],
        "mrope_sha256": None,
        "mrope_delta": None,
    }


def run(request, args):
    from pathlib import Path

    # vLLM's rule (`ModelConfig.uses_mrope`): M-RoPE when the config has an
    # `mrope_section`. Transformers may have no config class for the others.
    text = (Path(request["model_dir"]) / "config.json").read_text()
    mrope = '"mrope_section"' in text
    config = transformers.AutoConfig.from_pretrained(request["model_dir"]) if mrope else None
    processors, cases = {}, {}
    for case in request["cases"]:
        backend = case["backend"]
        if backend not in processors:
            processors[backend] = processor(request["model_dir"], backend)
        tokenizer = processors[backend].tokenizer
        prompt = tokenizer.decode(case["input_ids"])
        assert tokenizer.encode(prompt, add_special_tokens=False) == case["input_ids"], case["name"]
        data = [(Path(request["images_dir"]) / name).read_bytes() for name in case["images"]]
        try:
            out = process(processors[backend], case, prompt, data)
        except Exception as error:  # noqa: BLE001 -- the refusal is the recorded result
            cases[case["name"]] = {"rejects": reason(error)}
            continue
        cases[case["name"]] = {"output": record(processors[backend], config, out)}
    default = transformers.AutoProcessor.from_pretrained(request["model_dir"])
    return {
        "version": transformers.__version__,
        "source": SOURCE,
        "processor": type(default).__name__,
        "backend": default.image_processor.backend,
        "omits": [],
        "cases": cases,
    }
