import os
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, cast, Sequence

import numpy as np
import PIL.Image
import torch
import inspect
from einops import rearrange
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    rescale_noise_cfg,
)
from diffusers.pipelines.text_to_video_synthesis import TextToVideoSDPipelineOutput
from diffusers.schedulers import KarrasDiffusionSchedulers
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.utils.torch_utils import randn_tensor
import math
import threading
from torch.nn.modules import Module
from torch._utils import ExceptionWrapper
import copy
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.text_to_video_synthesis.pipeline_output import TextToVideoSDPipelineOutput
from dataclasses import dataclass, field


@dataclass
class APGonfigs:
    eta: float = 1.0
    norm_threshold: float = 10.0
    momentum: float = 0


def tensor2vid(video: torch.Tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) -> List[np.ndarray]:
    # This code is copied from https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/pipelines/multi_modal/text_to_video_synthesis_pipeline.py#L78
    # reshape to ncfhw
    mean = torch.tensor(mean, device=video.device).reshape(1, -1, 1, 1, 1)
    std = torch.tensor(std, device=video.device).reshape(1, -1, 1, 1, 1)
    # unnormalize back to [0,1]
    video = video.mul_(std).add_(mean)
    video.clamp_(0, 1)
    # prepare the final outputs
    i, c, f, h, w = video.shape
    images = video.permute(2, 3, 0, 4, 1).reshape(
        f, h, i * w, c
    )  # 1st (frames, h, batch_size, w, c) 2nd (frames, h, batch_size * w, c)
    images = images.unbind(dim=0)  # prepare a list of indvidual (consecutive frames)
    images = [(image.cpu().numpy() * 255).astype("uint8") for image in images]  # f h w c
    return images


class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average


def project(
    v0: torch.Tensor,  # [B, C, H, W]
    v1: torch.Tensor,  # [B, C, H, W]
):
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3, -4])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3, -4], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel

    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)


def normalized_guidance(
    pred_cond: torch.Tensor,  # [B, C, H, W]
    pred_uncond: torch.Tensor,  # [B, C, H, W]
    guidance_scale: float,
    momentum_buffer: MomentumBuffer = None,
    eta: float = 1.0,
    norm_threshold: float = 0.0,
):
    diff = pred_cond - pred_uncond
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average
    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=[-1, -2, -3, -4], keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor
    diff_parallel, diff_orthogonal = project(diff, pred_cond)
    normalized_update = diff_orthogonal + eta * diff_parallel
    pred_guided = pred_cond + (guidance_scale - 1) * normalized_update

    return pred_guided


def parallel_apply_mp(
    engine,
    modules: Sequence[Module],
    inputs: Sequence[Any],
    kwargs_tup: Optional[Sequence[Dict[str, Any]]] = None,
    devices: Optional[Sequence[Optional[Union[int, torch.device]]]] = None,
    iterative_inputs: bool = False,
) -> List[Any]:
    results = engine.generate(modules, inputs, kwargs_tup, devices)

    return results


class AugmentedTextToVideoCogPipeline(DiffusionPipeline, StableDiffusionMixin, TextualInversionLoaderMixin, LoraLoaderMixin):
    r"""
    Pipeline for text-to-video generation.
    """

    model_cpu_offload_seq = "text_encoder->unet->vae"

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder:CLIPTextModel,
        tokenizer: CLIPTokenizer,
        model,
        scheduler: KarrasDiffusionSchedulers,
        controlnet=None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            model=model,
            scheduler=scheduler,
        )
        try:
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        except:
            self.vae_scale_factor = 1

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        self.register_modules(
            controlnet=controlnet,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.conditioning_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=True,
        )

        self.use_SDXL = False
        self.use_fp32_vae = False
        self.use_motion_aesthetic_conditioning = False
        self.vae.enable_slicing()
        self.vae.enable_tiling()

        self.models = [
            copy.deepcopy(self.model).to(f"cuda:{rank}") for rank in range(1, min(2, torch.cuda.device_count()))
        ]
        self.devices = [torch.device("cuda:0"), torch.device("cuda:1")]

    def prepare_latent_model_input_and_timestep(
        self,
        latents,
        t,
        condition_t,
        num_frames,
        first_frame_latent=None,
        do_classifier_free_guidance=False,
        t_cond=0,
        ffcond_noise_level=0.3,
        cond_vid_latents=None,
        ffcond_rand_tensor=None,
    ):
        """
        Prepare the latent model input and timestep for the model. If doing classifier free guidance, the latents are
        duplicated. If conditioning on frames, the conditioning_mask is used to replace the latents for the
        conditioning frames, the timesteps are expanded to the number of frames and the timesteps for the conditioning
        frames are set to t_cond.
        """
        try:
            # scale if necessary
            latents = self.scheduler.scale_model_input(latents, t)
        except:
            pass

        condition_t = condition_t[None]

        # expand t if we are doing classifier free guidance
        condition_t = torch.cat([condition_t] * 2) if do_classifier_free_guidance else condition_t

        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

        if False: #self.model.first_frame_conditioned and first_frame_latent is not None:

            # weighting = 0.3
            noisy_first_frame_latent = (
                ffcond_noise_level * ffcond_rand_tensor + (1 - ffcond_noise_level) * first_frame_latent
            )

            from einops import repeat

            if cond_vid_latents is None:
                latent_model_input[:, :1, :16] = noisy_first_frame_latent
                latent_model_input[:, :, 16:] = 0
            else:
                second_part = repeat(
                    cond_vid_latents[:, 1:], "b f c h w -> (repeat b) f c h w", repeat=first_frame_latent.shape[0]
                )

                latent_model_input[:, :, 16:] = torch.cat([noisy_first_frame_latent, second_part], dim=1)
        else:
            if cond_vid_latents is not None:
                latent_model_input[:, :, 16:] = cond_vid_latents
            else:
                latent_model_input[:, :, 16:] = 0

        return latent_model_input, condition_t

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        text_encoder_max_length: int = 128,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        pooled_prompt_embeds = None
        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                max_length=text_encoder_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
                print(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
            )

            prompt_embeds = prompt_embeds[0]

        def duplicate_prompt_embds(prompt_embeds_in):
            def sub_duplicate_prompt_embds(prompt_embeds):
                prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
                bs_embed, seq_len, _ = prompt_embeds.shape

                # duplicate text embeddings for each generation per prompt, using mps friendly method
                prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
                prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

                return prompt_embeds

            if isinstance(prompt_embeds_in, list):
                return [sub_duplicate_prompt_embds(pe) for pe in prompt_embeds_in]

            return sub_duplicate_prompt_embds(prompt_embeds_in)

        prompt_embeds = duplicate_prompt_embds(prompt_embeds)

        negative_pooled_prompt_embeds = None
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            uncond_input = self.tokenizer(
                uncond_tokens,
                max_length=text_encoder_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
            )

            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            negative_prompt_embeds = duplicate_prompt_embds(negative_prompt_embeds)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if isinstance(prompt_embeds, list):
                prompt_embeds = [
                    torch.cat([negative_prompt_embed, prompt_embed])
                    for negative_prompt_embed, prompt_embed in zip(negative_prompt_embeds, prompt_embeds)
                ]
            else:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        shape = (
            batch_size,
            (num_frames // 4) + 1,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        try:
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
        except:
            pass

        return latents

    def decode_latents(self, latents: torch.Tensor, num_seconds: int):
        # latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        dim_f = latents.shape[1]
        latents = rearrange(latents[:, :, :16], "b f c h w -> b c f h w")
        # latents = 1 / self.vae.config.scaling_factor * latents

        # self.vae.num_latent_frames_batch_size = int(round((dim_f - 1) / 12))

        frames = self.vae.decode(latents).sample
        return frames

    def vae_encode(self, pixel_values):
        dim_b, _, dim_f = pixel_values.shape[:3]  # B C F H W

        with torch.no_grad():
            num_vae_calls = 0

            # self.vae._clear_fake_context_parallel_cache()
            # Make sure seq len must be 4K + 1
            assert dim_f == ((dim_f - 1) // 4 * 4) + 1
            import math

            sub_seq_len = 8

            num_step_size = sub_seq_len
            num_steps = max(int(math.ceil((dim_f - 1) / num_step_size)), 1)

            actual_sub_seq_len = min(sub_seq_len, dim_f)

            sub_batch_size = max(int(math.floor(dim_b / math.ceil(actual_sub_seq_len * dim_b / (sub_seq_len + 1)))), 1)

            starting_batch_idx = 0
            batch_latents_list = []

            batch_counter = 0
            while True:
                self.vae._clear_fake_context_parallel_cache()
                batched_pixel_values = pixel_values[starting_batch_idx : starting_batch_idx + sub_batch_size]

                counter = 0
                latents_list = []
                for i in range(num_steps):

                    if i == 0:
                        end_idx = num_step_size + 1
                        start_idx = 0
                        counter = end_idx
                    else:
                        start_idx = counter
                        end_idx = counter + num_step_size
                        counter = end_idx

                    # print(f"batched_pixel_values shape: {batched_pixel_values[:, :, start_idx:end_idx].shape}")
                    latents_list.append(self.vae.encode(batched_pixel_values[:, :, start_idx:end_idx]).latent_dist.mean)
                    num_vae_calls += 1
                    torch.cuda.empty_cache()

                batch_latents_list.append(torch.cat(latents_list, dim=2) if len(latents_list) > 1 else latents_list[0])
                starting_batch_idx += sub_batch_size
                if starting_batch_idx >= dim_b:
                    break

                batch_counter += 1

            # latents = rearrange(torch.cat(batch_latents_list, dim=0), "b c f h w -> b f c h w").to(
            #     dtype=self.half_dtype
            # )

        return torch.cat(batch_latents_list, dim=0)

    def get_resize_crop_region_for_grid(self, src, tgt_width, tgt_height):
        # Convert all inputs to tensors if they are not already
        if not isinstance(src, torch.Tensor):
            src = torch.tensor(src, dtype=torch.float32)
        if not isinstance(tgt_width, torch.Tensor):
            tgt_width = torch.tensor(tgt_width, dtype=torch.float32)
        if not isinstance(tgt_height, torch.Tensor):
            tgt_height = torch.tensor(tgt_height, dtype=torch.float32)
        
        # Unpack the source dimensions
        h, w = src[0], src[1]
        
        # Compute the resize dimensions
        r = h / w
        if r > (tgt_height / tgt_width):
            resize_height = tgt_height
            resize_width = torch.round(tgt_height / h * w)
        else:
            resize_width = tgt_width
            resize_height = torch.round(tgt_width / w * h)
        
        # Compute crop coordinates
        crop_top = torch.round((tgt_height - resize_height) / 2.0)
        crop_left = torch.round((tgt_width - resize_width) / 2.0)
        
        crop_bottom = crop_top + resize_height
        crop_right = crop_left + resize_width
        
        return (crop_top, crop_left), (crop_bottom, crop_right)

    def prepare_rope_ids(self, latent_shape, device, dtype=torch.bfloat16):
        from einops import repeat
        batch_size, num_frames, channels, height, width = latent_shape
        original_height = 256
        original_width = 448
        patch_size = 2
        grid_height = height // patch_size
        grid_width = width // patch_size

        base_size_width = original_width // (8 * patch_size)
        base_size_height = original_height // (8 * patch_size)

        grid_crops_coords = self.get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        start, stop = grid_crops_coords

        img_ids = torch.zeros(
            num_frames, height // 2, width // 2, 3, dtype=dtype, device=device
        )
        img_ids[..., 0] = (
            img_ids[..., 0]
            + torch.arange(num_frames, dtype=dtype, device=device)[:, None, None]
        )
        step_size_h = (stop[0] - start[0]) / (height // 2)
        img_ids[..., 1] = (
            img_ids[..., 1]
            + torch.arange(start[0], stop[0], step_size_h, dtype=dtype, device=device)[
                : (height // 2), None
            ]
        )
        step_size_w = (stop[1] - start[1]) / (width // 2)
        img_ids[..., 2] = (
            img_ids[..., 2]
            + torch.arange(start[1], stop[1], step_size_w, dtype=dtype, device=device)[
                None, : (width // 2)
            ]
        )
        img_ids = repeat(img_ids, "n h w c -> b (n h w) c", b=batch_size)

        txt_ids = torch.zeros(
            batch_size, 226, 3, dtype=dtype, device=device
        )

        return img_ids, txt_ids
    
    

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 16,
        image: Optional[
            Union[
                torch.FloatTensor,
                PIL.Image.Image,
                np.ndarray,
                List[torch.FloatTensor],
                List[PIL.Image.Image],
                List[np.ndarray],
            ]
        ] = None,
        video_frames: Union[np.ndarray, List[np.ndarray], List[PIL.Image.Image]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 9.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        frame_every: Union[torch.Tensor, int] = 2,
        guidance_rescale: float = 0.0,
        use_long_prompts: bool = True,
        cfg_stop_point: float = 1.0,
        seg_cfg: bool = True,
        fps: int = 8,
        ffcond_noise_level: float = 0.3,
        apg_config: APGonfigs = None,
        ff_cond_scheduler: str = "0.1",
        strength: float = 0.3,
        multidiff: bool = False,
        window_len: int = 7,
        hop_size: int = 1,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated video.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated video.
            num_frames (`int`, *optional*, defaults to 16):
                The number of video frames that are generated. Defaults to 16 frames which at 8 frames per seconds
                amounts to 2 seconds of video.
            image (Optional[
                        Union[
                            torch.FloatTensor,
                            PIL.Image.Image,
                            np.ndarray,
                            List[torch.FloatTensor],
                            List[PIL.Image.Image],
                            List[np.ndarray],
                        ]
                    ] = None,):
                The image input for first frame conditioning.
            video_frames (`np.ndarray` or `List[np.ndarray]` or `List[PIL.Image.Image]`):
                `video` frames representing a video batch, that will be used as the starting point for the
                process.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality videos at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`. Latents should be of shape
                `(batch_size, num_channel, num_frames, height, width)`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated video. Choose between `torch.FloatTensor` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.text_to_video_synthesis.TextToVideoSDPipelineOutput`] instead
                of a plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            conditioning_indices (`List[List[int]]`, *optional*):
                A list of lists of integers. Each list of integers corresponds to the indices of the frames that
                should be conditioned on. If not specified, the model is not conditioned on any frame
            cfg_stop_point (`float`, *optional*, defaults to 1.0):
                Percentage of inference steps after which we stop using classifier free guidence. This is because
                CFG is not really necessary towards the latter half of denoising and turning CFG off saves compute.
                Recommended value: 0.4.
            seg_cfg (`bool`, *optional*, defaults to `True`):
                Whether to use segregated classifier free guidance. Segregated CFG reduces the batch size to 1.
            looping (`bool`, *optional*, defaults to `False`):
                If True, the generated video will be conditioned to loop. This is done by generating a video that
                is num_frames + 1 long, where the latents for the last frame are conditioned to be the same as the
                first frame. We only return the first num_frames frames of the generated video.


        Examples:

        Returns:
            [`~pipelines.text_to_video_synthesis.TextToVideoSDPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.text_to_video_synthesis.TextToVideoSDPipelineOutput`] is
                returned, otherwise a `tuple` is returned where the first element is a list with the generated frames.
        """

        num_images_per_prompt = 1


        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        try:
            device = self._execution_device
        except AttributeError:
            # device = "cuda"
            device = torch.device("cuda:0")

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        (
            prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            text_encoder_max_length=226,
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        conditioning_frames = None
        conditioning_frames_latent = None
        if image is not None:
            conditioning_frames = self.prepare_conditioning_image(
                image,
                batch_size=batch_size,
                height=height,
                width=width,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=self.vae.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=False,
            )
            conditioning_frames_latent = self.vae.encode(
                rearrange(conditioning_frames, "B (C F) H W -> B C F H W", F=1)
            ).latent_dist.mean
            # conditioning_frames_latent *= self.vae.config.scaling_factor
            conditioning_frames_latent = rearrange(conditioning_frames_latent, "B C F H W -> B F C H W")

            # # Add small amount of noise to conditioning frame to align with training
            # noise = torch.randn_like(conditioning_frames_latent)
            # noised_conditioning_frames_latent = self.scheduler.add_noise(
            #     conditioning_frames_latent.to(dtype=torch.float32),
            #     noise.to(dtype=torch.float32),
            #     torch.zeros(noise.shape[0], dtype=torch.float32, device=noise.device),
            # ).to(dtype=conditioning_frames_latent.dtype)

        cond_vid_latents = None
        do_video_upscaling = False
        
        
        num_channels_latents = self.model.config.in_channels

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            num_frames,
            height,
            width,
            torch.bfloat16,
            device,
            generator,
            latents,
        )


        img_ids, txt_ids = self.prepare_rope_ids(latents.shape, device=device)

        momentum_buffer = None
        if apg_config is not None:
            momentum_buffer = MomentumBuffer(momentum=apg_config.momentum)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        multidiff_count = torch.zeros_like(latents[:, :, :16])
        multidiff_val = torch.zeros_like(latents[:, :, :16])
        max_frame_latent = latents.shape[1]
        md_list = []

        if multidiff:
            for i in range(math.ceil((max_frame_latent - window_len) / hop_size) + 1):
                if i * hop_size + window_len > max_frame_latent:
                    md_list.append((max_frame_latent - window_len, max_frame_latent))
                else:
                    md_list.append((i * hop_size, i * hop_size + window_len))
        else:
            md_list.append((0, max_frame_latent))
        print("md_list", md_list)

        ffcond_rand_tensor = None
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                multidiff_count.zero_()
                multidiff_val.zero_()
                for md_start, md_end in md_list:
                    latents_chunk = latents[:, md_start:md_end]
                    cond_vid_latents_chunk = (
                        cond_vid_latents[:, md_start:md_end] if cond_vid_latents is not None else None
                    )
                    if i >= int(cfg_stop_point * len(timesteps)):
                        do_classifier_free_guidance = False

                    conditioning = t

                    # used to change the inputs when we stop cfg after cfg_stop_point
                    reduce_tensor = prompt_embeds.shape[0] > batch_size and not do_classifier_free_guidance

                    encoder_hidden_states_inputs = prompt_embeds[batch_size:] if reduce_tensor else prompt_embeds

                    if md_start == 0:
                        first_frame_latent = (
                            conditioning_frames_latent[batch_size:]
                            if conditioning_frames_latent is not None and reduce_tensor
                            else conditioning_frames_latent
                        )
                    else:
                        first_frame_latent = None

                    multiplier_func = lambda t: eval(ff_cond_scheduler)
                    multiplier = multiplier_func(t)

                    if (
                        first_frame_latent is not None
                        and ffcond_rand_tensor is None
                    ):
                        ffcond_rand_tensor = randn_tensor(
                            first_frame_latent[:1].shape,
                            generator=generator,
                            device=first_frame_latent.device,
                            dtype=first_frame_latent.dtype,
                        )

                    # prepare the latent model input and timestep
                    latent_model_input, conditioning = self.prepare_latent_model_input_and_timestep(
                        latents_chunk,
                        t,
                        conditioning,
                        num_frames,
                        first_frame_latent=first_frame_latent,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        cond_vid_latents=cond_vid_latents_chunk,
                        ffcond_noise_level=multiplier,
                        ffcond_rand_tensor=ffcond_rand_tensor,
                    )

                    conditioning = conditioning.to(
                        latent_model_input
                    )  # ensure conditioning has the same dtype as the model input

                    # timestep_cond = None
                    # if do_video_upscaling:
                    #     timestep_cond = strength * torch.ones_like(conditioning)
                    # elif self.model.first_frame_conditioned and first_frame_latent is not None:
                    #     timestep_cond = multiplier * torch.ones_like(conditioning)
                    # elif self.model.first_frame_conditioned:
                    #     timestep_cond = conditioning

                    # latent_model_input = latent_model_input.view(2, 13, 32, 45, 2, 80, 2)
                    # latent_model_input = latent_model_input.permute(0, 1, 3, 5, 2, 4, 6)
                    # latent_model_input = latent_model_input.reshape(2, 46800, 128)

                    # predict the noise residual
                    noise_pred = self.model(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=encoder_hidden_states_inputs,
                        timestep=conditioning,
                        # timestep_cond=None,
                        img_ids=img_ids,
                        txt_ids=txt_ids,
                    )[0]

                    # noise_pred = noise_pred.reshape(2, 13, 45, 80, 16, 2, 2)
                    # noise_pred = noise_pred.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
                    # noise_pred = rearrange(noise_pred, "b (f h w) c -> b f c h w", f=13, h=45, w=80)

                    # NaNs checker.
                    if torch.isnan(noise_pred.reshape(-1)).sum().item() > 0:
                        raise ValueError(f"`noise_pred` has NaNs!!!")

                    # perform guidance
                    noise_pred_uncond, noise_pred_text = None, None
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

                        # print(f"eta: {apg_config.eta}   eta==1: {apg_config.eta == 1.0}")

                        if momentum_buffer is None or apg_config.eta == 1.0:
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                        else:
                            # x0_pred = normalized_guidance(
                            #     self.scheduler.convert_model_output(noise_pred_text, sample=latents[:, :, :16]),
                            #     self.scheduler.convert_model_output(noise_pred_uncond, sample=latents[:, :, :16]),
                            #     guidance_scale,
                            #     momentum_buffer,
                            #     eta=apg_config.eta,
                            #     norm_threshold=apg_config.norm_threshold,
                            # )

                            # noise_pred = self.scheduler.compute_model_prediction(
                            #     latents[:, :, :16].to(dtype=torch.float32), x0_pred.to(dtype=torch.float32)
                            # ).to(dtype=noise_pred.dtype)

                            noise_pred = normalized_guidance(
                                noise_pred_text,
                                noise_pred_uncond,
                                guidance_scale,
                                momentum_buffer,
                                eta=apg_config.eta,
                                norm_threshold=apg_config.norm_threshold,
                            )

                    if do_classifier_free_guidance and guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                    if md_start == 0:
                        multidiff_val[:, md_start:md_end] += noise_pred
                        multidiff_count[:, md_start:md_end] += 1
                    else:
                        multidiff_val[:, md_start + 1 : md_end] += noise_pred[:, 1:]
                        multidiff_count[:, md_start + 1 : md_end] += 1

                noise_pred = multidiff_val / multidiff_count
                latents[:, :, :16] = self.scheduler.step(
                    noise_pred[:, :, :16].to(dtype=torch.float32),
                    t,
                    latents[:, :, :16].to(dtype=torch.float32),
                ).prev_sample.to(dtype=noise_pred.dtype)

                # if self.model.first_frame_conditioned and first_frame_latent is not None and not do_video_upscaling:
                #     latents[:, :1, :16] = first_frame_latent[:1]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # Add first frame back in
        # if first_frame_latent is not None and not isinstance(self.vae, Haiper3DVAE):
        #     latents[:, :, 0] = conditioning_frames_latent[0]

        if output_type == "latent":
            return TextToVideoSDPipelineOutput(frames=latents)

        # delete used local variables to free gpu memory
        del (
            conditioning,
            conditioning_frames,
            latent_model_input,
            noise_pred,
            noise_pred_text,
            noise_pred_uncond,
            prompt_embeds,
            timesteps,
        )
        if video_frames is not None:
            del video_normalized_torch

        torch.cuda.empty_cache()

        video_tensor = self.decode_latents(latents.to(self.vae.dtype), num_frames // fps)

        if output_type == "pt":
            video = video_tensor
        else:
            # video = pipeline_text_to_video_synth.tensor2vid(video_tensor)
            video = tensor2vid(video_tensor)

        # Offload last model to CPU
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return TextToVideoSDPipelineOutput(frames=video)
