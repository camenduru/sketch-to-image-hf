"""
 * Copyright (c) 2023 Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: Apache License 2.0
 * For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/
 * By Can Qin
 * Modified from ControlNet repo: https://github.com/lllyasviel/ControlNet
 * Copyright (c) 2023 Lvmin Zhang and Maneesh Agrawala
"""

import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import os

from annotator.util import resize_image, HWC3
from utils import create_model
from lib.ddim_hacked import DDIMSampler

from safetensors.torch import load_file as stload
from collections import OrderedDict
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
)
refiner.to("cuda")


model = create_model("./models/cldm_v15_unicontrol.yaml").cpu()
model_url = "https://huggingface.co/Robert001/UniControl-Model/resolve/main/unicontrol_v1.1.st"

ckpts_path = "./"
# model_path = os.path.join(ckpts_path, "unicontrol_v1.1.ckpt")
model_path = os.path.join(ckpts_path, "unicontrol_v1.1.st")

if not os.path.exists(model_path):
    from basicsr.utils.download_util import load_file_from_url

    load_file_from_url(model_url, model_dir=ckpts_path)

model_dict = OrderedDict(stload(model_path, device="cpu"))
model.load_state_dict(model_dict, strict=False)
# model.load_state_dict(load_state_dict(model_path, location='cuda'), strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process_sketch(
    input_image,
    prompt,
    a_prompt,
    n_prompt,
    num_samples,
    ddim_steps,
    guess_mode,
    strength,
    scale,
    seed,
    eta,
):
    with torch.no_grad():
        input_image = np.array(input_image)
        # print all unique values of array
        img = 255 - input_image
        H, W, C = img.shape

        detected_map = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        # seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        task_dic = {}
        task_dic["name"] = "control_hedsketch"
        task_instruction = "sketch to image"
        task_dic["feature"] = model.get_learned_conditioning(task_instruction)[:, :1, :]

        cond = {
            "c_concat": [control],
            "c_crossattn": [model.get_learned_conditioning([prompt + ", " + a_prompt] * num_samples)],
            "task": task_dic,
        }

        un_cond = {
            "c_concat": [control * 0] if guess_mode else [control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)],
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)
        model.control_scales = (
            [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        )
        samples, intermediates = ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond,
        )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (
            (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
            .cpu()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )

        result_image = [x_samples[i] for i in range(num_samples)][0]
        result_image = Image.fromarray(result_image)
        generator = torch.Generator("cuda").manual_seed(seed)
        results = [result_image] + [refiner(prompt=prompt, generator=generator, image=result_image).images[0]]

    return results


demo = gr.Blocks()
with demo:
    # gr.Markdown("## UniControl Stable Diffusion with Sketch Maps")
    # input_image = gr.Image(source="upload", type="numpy", tool="sketch")
    with gr.Row():
        input_image = gr.Sketchpad(
            shape=(512, 512), tool="pencil", brush_radius=6, type="pil", image_mode="RGB"
        ).style(height=512, width=512)
        # input_image = gr.Image(source="upload", type="numpy")
        result_gallery = gr.Gallery(label="Output", show_label=False, elem_id="gallery").style(
            grid=2, height=512, width=512
        )
    prompt = gr.Textbox(label="Prompt")
    run_button = gr.Button(label="Run")
    with gr.Accordion("Advanced options", open=False):
        num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
        strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
        guess_mode = gr.Checkbox(label="Guess Mode", value=False)
        detect_resolution = gr.Slider(label="HED Resolution", minimum=128, maximum=1024, value=512, step=1)
        ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=35, step=1)
        scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
        seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
        eta = gr.Number(label="eta (DDIM)", value=0.0)
        a_prompt = gr.Textbox(label="Added Prompt", value="best quality, extremely detailed")
        n_prompt = gr.Textbox(
            label="Negative Prompt",
            value="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
        )
    ips = [
        input_image,
        prompt,
        a_prompt,
        n_prompt,
        num_samples,
        ddim_steps,
        guess_mode,
        strength,
        scale,
        seed,
        eta,
    ]
    run_button.click(fn=process_sketch, inputs=ips, outputs=[result_gallery])

demo.launch(server_name="0.0.0.0")
