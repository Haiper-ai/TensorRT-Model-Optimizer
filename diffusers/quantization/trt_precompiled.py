# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse

import torch
from config import DYNAMIC_SHAPES, get_io_shapes, remove_nesting, update_dynamic_axes
from onnx_utils.export import AXES_NAME, generate_dummy_inputs
from pipe import AugmentedTextToVideoCogPipeline
import modelopt.torch.opt as mto
from modelopt.torch._deploy._runtime import RuntimeRegistry
from modelopt.torch._deploy.device_model import DeviceModel
from modelopt.torch._deploy.utils import get_onnx_bytes_and_metadata
from quantize import load_deepspeed_checkpoint
from schedulers import DPMSolverMultistepSchedulerRF
import yaml
from model import QuantisedCogVideoXTransformer3DModel
from dotmap import DotMap
import imageio

def save_video(video, output_path, output_fps=None):
    """Encode mp4 with imageio[ffmpeg] that you can view directly on VSCode instead of VLC."""

    if output_fps is None:
        raise ValueError("You are saving video with `imageio`. Plz set output_fps!")
    output_path = str(output_path)

    imageio.mimwrite(output_path, video, quality=8, format="mp4", fps=output_fps)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="cog",
        choices=[
            "sdxl-1.0",
            "sdxl-turbo",
            "sd2.1",
            "sd2.1-base",
            "sd3-medium",
            "flux-dev",
            "flux-schnell",
            "cog",
        ],
    )
    parser.add_argument("--pretrained_model_state_dict", type=str, default=None)
    parser.add_argument("--pretrained_model_config", type=str, default=None)
    parser.add_argument("--restore-from", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="A cat holding a sign that says hello world")
    parser.add_argument("--onnx-load-path", type=str, default="")
    parser.add_argument("--trt-engine-path", type=str, default=None)
    args = parser.parse_args()


    with open(args.pretrained_model_config, "r") as file:
        model_config = DotMap(yaml.safe_load(file))

    (
        tokenizer,
        text_encoder,
        vae,
        backbone,
    ) = load_deepspeed_checkpoint(
        model_config,
        args.pretrained_model_state_dict,
    )

    if args.restore_from:
        mto.restore(backbone, args.restore_from)

    backbone.to(device="cuda", dtype=torch.bfloat16)

    # Generate dummy inputs for the Flux model
    dummy_inputs = generate_dummy_inputs(args.model, "cuda", True)

    # Define dynamic axes for ONNX export
    dynamic_axes = {
        "hidden_states": {0: "batch_size"},
        "encoder_hidden_states": {0: "batch_size"},
        "timestep": {0: "batch_size"},
        # "timestep_cond": {0: "batch_size"},
        "img_ids": {0: "batch_size"},
        "txt_ids": {0: "batch_size"},
        "out.0": {0: "batch_size"},
    }

    # Postprocess the dynamic axes to match the input and outputs names with DeviceModel
    if args.onnx_load_path == "":
        update_dynamic_axes(args.model, dynamic_axes)

    compilation_args = DYNAMIC_SHAPES[args.model]

    # We only need to remove the nesting for SDXL models as they contain the nested input added_cond_kwargs
    # which are renamed by the DeviceModel
    if args.onnx_load_path != "" and args.model in ["sdxl-1.0", "sdxl-turbo"]:
        remove_nesting(compilation_args)

    # Define deployment configuration
    deployment = {
        "runtime": "TRT",
        "version": "10.3",
        "precision": "stronglyTyped",
        "onnx_opset": "17",
    }

    client = RuntimeRegistry.get(deployment)

    # Export onnx model and get some required names from it
    onnx_bytes, metadata = get_onnx_bytes_and_metadata(
        model=backbone,
        dummy_input=dummy_inputs,
        onnx_load_path=args.onnx_load_path,
        dynamic_axes=dynamic_axes,
        onnx_opset=int(deployment["onnx_opset"]),
        remove_exported_model=False,
    )

    if not args.trt_engine_path:
        # Compile the TRT engine from the exported ONNX model
        compiled_model = client.ir_to_compiled(onnx_bytes, compilation_args)
    else:
        with open(args.trt_engine_path, "rb") as f:
            compiled_model = f.read()

    # The output shapes will need to be specified for models with dynamic output dimensions
    device_model = DeviceModel(
        client,
        compiled_model,
        metadata,
        compilation_args,
        {},
    )

    
    inference_scheduler = DPMSolverMultistepSchedulerRF.from_config(model_config.inferece_scheduler_config)

    pipe = AugmentedTextToVideoCogPipeline(
        scheduler=inference_scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        model=device_model
    )

    # Move the backbone back to the CPU and set the transformer to the compiled device model
    backbone.to("cpu")
    pipe.to("cuda")

    video = pipe(
        prompt="A dog running",
        num_inference_steps=20,
        height=720,
        width=1280,
        guidance_scale=6,
        num_frames=49,
    ).frames

    output_path = save_video(video=video, output_path="test.mp4", output_fps=8)

    # image.save(f"{args.model}.png")
    # print(f"Image generated using {args.model} model saved as {args.model}.png")

    print(f"Inference latency of the backbone of the pipeline is {device_model.get_latency()} ms")


if __name__ == "__main__":
    main()
