import torch
from typing import List, Union, Optional, Dict, Any, Callable, Type, Tuple

from diffusers.pipelines import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    StableDiffusion3PipelineOutput,
    SD3Transformer2DModel,
    calculate_shift,
    retrieve_timesteps,
)
from diffusers.models.attention_processor import Attention, F
from diffusers.models.embeddings import apply_rotary_emb
from transformers import pipeline
import numpy as np
from peft.tuners.tuners_utils import BaseTunerLayer
from accelerate.utils import is_torch_version

from contextlib import contextmanager

import cv2

from PIL import Image, ImageFilter


def seed_everything(seed: int = 42):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)


def clip_hidden_states(hidden_states: torch.FloatTensor) -> torch.FloatTensor:
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
    return hidden_states


def encode_images(pipeline: StableDiffusion3Pipeline, images: torch.Tensor):
    """
    Encodes the images into tokens and ids for sd3 pipeline.
    """
    input_device = pipeline.device
    images = pipeline.image_processor.preprocess(images)
    images = images.to(input_device).to(pipeline.dtype)
    images = pipeline.vae.encode(images).latent_dist.sample()
    images = (
        images - pipeline.vae.config.shift_factor
    ) * pipeline.vae.config.scaling_factor


    return images


depth_pipe = None


def convert_to_condition(
    condition_type: str,
    raw_img: Union[Image.Image, torch.Tensor],
    blur_radius: Optional[int] = 5,
) -> Union[Image.Image, torch.Tensor]:
    if condition_type == "depth":
        global depth_pipe
        depth_pipe = depth_pipe or pipeline(
            task="depth-estimation",
            model="LiheYoung/depth-anything-small-hf",
            device="cpu",  # Use "cpu" to enable parallel processing
        )
        source_image = raw_img.convert("RGB")
        condition_img = depth_pipe(source_image)["depth"].convert("RGB")
        return condition_img
    elif condition_type == "canny":
        img = np.array(raw_img)
        edges = cv2.Canny(img, 100, 200)
        edges = Image.fromarray(edges).convert("RGB")
        return edges
    elif condition_type == "coloring":
        return raw_img.convert("L").convert("RGB")
    elif condition_type == "deblurring":
        condition_image = (
            raw_img.convert("RGB")
            .filter(ImageFilter.GaussianBlur(blur_radius))
            .convert("RGB")
        )
        return condition_image
    else:
        print("Warning: Returning the raw image.")
        return raw_img.convert("RGB")


class Condition(object):
    def __init__(
        self,
        condition: Union[Image.Image, torch.Tensor],
        adapter_setting: Union[str, dict],
        position_delta=None,
        position_scale=1.0,
        latent_mask=None,
        is_complement=False,
    ) -> None:
        self.condition = condition
        self.adapter = adapter_setting
        self.position_delta = position_delta
        self.position_scale = position_scale
        self.latent_mask = (
            latent_mask.T.reshape(-1) if latent_mask is not None else None
        )
        self.is_complement = is_complement

    def encode(
        self, pipe: StableDiffusion3Pipeline, empty: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        
        # condition_empty = Image.new("RGB", self.condition.size, (0, 0, 0))
        condition_empty = Image.new("RGB", self.condition.shape[-2:], (0, 0, 0))
        tokens = encode_images(pipe, condition_empty if empty else self.condition)

        if self.latent_mask is not None:
            tokens = tokens[:, self.latent_mask]

        return tokens


@contextmanager
def specify_lora(lora_modules: List[BaseTunerLayer], specified_lora):
    # Filter valid lora modules
    valid_lora_modules = [m for m in lora_modules if isinstance(m, BaseTunerLayer)]
    # Save original scales
    original_scales = [
        {
            adapter: module.scaling[adapter]
            for adapter in module.active_adapters
            if adapter in module.scaling
        }
        for module in valid_lora_modules
    ]
    # Enter context: adjust scaling
    for module in valid_lora_modules:
        for adapter in module.active_adapters:
            if adapter in module.scaling:
                module.scaling[adapter] = 1 if adapter == specified_lora else 0
    try:
        yield
    finally:
        # Exit context: restore original scales
        for module, scales in zip(valid_lora_modules, original_scales):
            for adapter in module.active_adapters:
                if adapter in module.scaling:
                    module.scaling[adapter] = scales[adapter]


def attn_forward(
    attn: Attention,
    hidden_states: List[torch.FloatTensor],
    adapters: List[str],
    hidden_states2: Optional[List[torch.FloatTensor]] = None,
    position_embs: Optional[List[torch.Tensor]] = None,
    group_mask: Optional[torch.Tensor] = None,
    cache_mode: Optional[str] = None,
    # to determine whether to cache the keys and values for this branch
    to_cache: Optional[List[torch.Tensor]] = None,
    cache_storage: Optional[List[torch.Tensor]] = None,
    **kwargs: dict,
) -> torch.FloatTensor:
    bs, _, _ = hidden_states[0].shape
    h2_n = len(hidden_states2) if hidden_states2 is not None else 0
    len_i = len(hidden_states)
    queries, keys, values = [], [], []

    # Prepare query, key, value for each encoder hidden state (text branch)
    if hidden_states2 is not None:
        for i, hidden_state in enumerate(hidden_states2):
            query = attn.add_q_proj(hidden_state)
            key = attn.add_k_proj(hidden_state)
            value = attn.add_v_proj(hidden_state)

            head_dim = key.shape[-1] // attn.heads
            reshape_fn = lambda x: x.view(bs, -1, attn.heads, head_dim).transpose(1, 2)

            query, key, value = map(reshape_fn, (query, key, value))
            query, key = attn.norm_added_q(query), attn.norm_added_k(key)

            queries.append(query)
            keys.append(key)
            values.append(value)

    else: 
        adapters = adapters[-len_i:]
        group_mask = group_mask[-len_i:,-len_i:]

    # Prepare query, key, value for each hidden state (image branch)
    for i, hidden_state in enumerate(hidden_states):
        with specify_lora((attn.to_q, attn.to_k, attn.to_v), adapters[i + h2_n]):
            query = attn.to_q(hidden_state)
            key = attn.to_k(hidden_state)
            value = attn.to_v(hidden_state)

        head_dim = key.shape[-1] // attn.heads
        reshape_fn = lambda x: x.view(bs, -1, attn.heads, head_dim).transpose(1, 2)

        query, key, value = map(reshape_fn, (query, key, value))
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        queries.append(query)
        keys.append(key)
        values.append(value)

    # Apply rotary embedding
    # if position_embs is not None:
    #     queries = [apply_rotary_emb(q, position_embs[i]) for i, q in enumerate(queries)]
    #     keys = [apply_rotary_emb(k, position_embs[i]) for i, k in enumerate(keys)]

    # if cache_mode == "write":
    #     for i, (k, v) in enumerate(zip(keys, values)):
    #         if to_cache[i]:
    #             cache_storage[attn.cache_idx][0].append(k)
    #             cache_storage[attn.cache_idx][1].append(v)

    attn_outputs = []
    for i, query in enumerate(queries):
        keys_, values_ = [], []
        # Add keys and values from other branches
        for j, (k, v) in enumerate(zip(keys, values)):
            if (group_mask is not None) and not (group_mask[i][j].item()):
                continue
            keys_.append(k)
            values_.append(v)
        # if cache_mode == "read":
        #     keys_.extend(cache_storage[attn.cache_idx][0])
        #     values_.extend(cache_storage[attn.cache_idx][1])
        # Add keys and values from cache TODO
        # Attention computation
        attn_output = F.scaled_dot_product_attention(
            query, torch.cat(keys_, dim=2), torch.cat(values_, dim=2)
        ).to(query.dtype)
        attn_output = attn_output.transpose(1, 2).reshape(bs, -1, attn.heads * head_dim)
        attn_outputs.append(attn_output)

    # Reshape attention output to match the original hidden states
    h_out, h2_out = [], []
    if hidden_states2 is not None:
        for i, hidden_state in enumerate(hidden_states2):
            if not attn.context_pre_only:
                h2_out.append(attn.to_add_out(attn_outputs[i]))
            else:
                h2_out.append(attn_outputs[i])

    for i, hidden_state in enumerate(hidden_states):
        h = attn_outputs[i + h2_n]
        if getattr(attn, "to_out", None) is not None:
            with specify_lora((attn.to_out[0],), adapters[i + h2_n]):
                h = attn.to_out[0](h)
        h_out.append(h)

    return (h_out, h2_out) if h2_n else h_out


def block_forward(
    self,
    image_hidden_states: List[torch.FloatTensor],
    text_hidden_states: List[torch.FloatTensor],
    tembs: List[torch.FloatTensor],
    adapters: List[str],
    attn_forward=attn_forward,
    **kwargs: dict,
):
    txt_n = len(text_hidden_states)

    img_variables, txt_variables = [], []

    for i, text_h in enumerate(text_hidden_states):
        if self.context_pre_only:
            txt_variables.append(self.norm1_context(text_h, tembs[i]))
        else:
            txt_variables.append(self.norm1_context(text_h, emb=tembs[i]))

    for i, image_h in enumerate(image_hidden_states):
        with specify_lora((self.norm1.linear,), adapters[i + txt_n]):
            img_variables.append(self.norm1(image_h, emb=tembs[i + txt_n]))

    # Attention.

    img_attn_output, txt_attn_output = attn_forward(
        self.attn,
        hidden_states=[each[0] for each in img_variables],
        hidden_states2=[each for each in txt_variables] if self.context_pre_only else [each[0] for each in txt_variables],
        adapters=adapters,
        **kwargs,
    )

    if self.use_dual_attention:
        img_attn_output2 = attn_forward(
        self.attn2,
        hidden_states=[each[5] for each in img_variables],
        adapters=adapters,
        **kwargs,
    )

    image_out = []
    for i in range(len(image_hidden_states)):
        if self.use_dual_attention:
            _, gate_msa, shift_mlp, scale_mlp, gate_mlp, _, gate_msa2  = img_variables[i]
        else:
            _, gate_msa, shift_mlp, scale_mlp, gate_mlp = img_variables[i]
        image_h = (
            image_hidden_states[i] + img_attn_output[i] * gate_msa.unsqueeze(1)
        ).to(image_hidden_states[i].dtype)
        if self.use_dual_attention:
            image_h = image_h + img_attn_output2[i] * gate_msa2.unsqueeze(1)

        norm_h = self.norm2(image_h) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        with specify_lora((self.ff.net[2],), adapters[i + txt_n]):
            if self._chunk_size is not None:
                image_h = image_h + self._chunked_feed_forward(self.ff, norm_h, self._chunk_dim, self._chunk_size) * gate_mlp.unsqueeze(1)
            else:
                image_h = image_h + self.ff(norm_h) * gate_mlp.unsqueeze(1)
        
        image_out.append(clip_hidden_states(image_h))


    text_out = []
    for i in range(len(text_hidden_states)):
        if self.context_pre_only:
            text_out.append(None)
        else:
            _, gate_msa, shift_mlp, scale_mlp, gate_mlp = txt_variables[i]
            text_h = text_hidden_states[i] + txt_attn_output[i] * gate_msa.unsqueeze(1)
            norm_h = (
                self.norm2_context(text_h) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            )
            if self._chunk_size is not None:
                text_h = chunked_feed_forward(self.ff_context, norm_h, self._chunk_dim, self._chunk_size) * gate_mlp.unsqueeze(1) + text_h
            else:
                text_h = self.ff_context(norm_h) * gate_mlp.unsqueeze(1) + text_h
            text_out.append(clip_hidden_states(text_h))

    return image_out, text_out




def transformer_forward(
    transformer: SD3Transformer2DModel,
    image_features: List[torch.Tensor],
    text_features: List[torch.Tensor] = None,
    pooled_projections: List[torch.Tensor] = None,
    timesteps: List[torch.LongTensor] = None,
    adapters: List[str] = None,
    # Assign the function to be used for the forward pass
    block_forward=block_forward,
    attn_forward=attn_forward,
    **kwargs: dict,
):
    self = transformer
    txt_n = len(text_features) if text_features is not None else 0

    adapters = adapters or [None] * (txt_n + len(image_features))
    assert len(adapters) == len(timesteps)

    # Preprocess the image_features
    height, width = image_features[0].shape[-2:]
    image_hidden_states = []
    for i, image_feature in enumerate(image_features):
        with specify_lora((self.pos_embed.proj,), adapters[i + txt_n]):
            image_hidden_states.append(self.pos_embed(image_feature))

    # Preprocess the text_features
    text_hidden_states = []
    for text_feature in text_features:
        text_hidden_states.append(self.context_embedder(text_feature))

    # Prepare embeddings of (timestep, guidance, pooled_projections)
    assert len(timesteps) == len(image_features) + len(text_features)
    def get_temb(timestep, pooled_projection):
        timestep = timestep.to(image_hidden_states[0].dtype) * 1000
        return self.time_text_embed(timestep, pooled_projection)
    tembs = [get_temb(*each) for each in zip(timesteps, pooled_projections)]

    # Prepare the gradient checkpointing kwargs
    gckpt_kwargs: Dict[str, Any] = (
        {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
    )

    # dual branch blocks
    for block in self.transformer_blocks:
        block_kwargs = {
            "self": block,
            "image_hidden_states": image_hidden_states,
            "text_hidden_states": text_hidden_states,
            "tembs": tembs,
            "adapters": adapters,
            "attn_forward": attn_forward,
            **kwargs,
        }
        if self.training and self.gradient_checkpointing:
            image_hidden_states, text_hidden_states = torch.utils.checkpoint.checkpoint(
                block_forward, **block_kwargs, **gckpt_kwargs
            )
        else:
            image_hidden_states, text_hidden_states = block_forward(**block_kwargs)


    image_hidden_states = self.norm_out(image_hidden_states[0], tembs[0])
    hidden_states = self.proj_out(image_hidden_states)

    # unpatchify
    patch_size = self.config.patch_size
    height = height // patch_size
    width = width // patch_size

    hidden_states = hidden_states.reshape(
        shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
    )
    hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
    output = hidden_states.reshape(
        shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
    )


    return (output,)


@torch.no_grad()
def generate(
    pipeline: StableDiffusion3Pipeline,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = 512,
    width: Optional[int] = 512,
    num_inference_steps: int = 28,
    timesteps: List[int] = None,
    guidance_scale: float = 3.5,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    # Condition Parameters (Optional)
    main_adapter: Optional[List[str]] = None,
    conditions: List[Condition] = [],
    image_guidance_scale: float = 1.0,
    transformer_kwargs: Optional[Dict[str, Any]] = {},
    kv_cache=False,
    latent_mask=None,
    **params: dict,
):
    self = pipeline

    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        prompt_3,
        height,
        width,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs

    # Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    # Prepare prompt embeddings
    (
        prompt_embeds, _,
        pooled_prompt_embeds, _
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_3=prompt_3,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )

    # Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels 
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    if latent_mask is not None:
        latent_mask = latent_mask.T.reshape(-1)
        latents = latents[:, latent_mask]

    # Prepare conditions
    c_latents, uc_latents, c_timesteps = ([], [], [])
    c_projections, c_adapters = ([], [])
    complement_cond = None
    for condition in conditions:
        tokens = condition.encode(self)
        c_latents.append(tokens)  # [batch_size, token_n, token_dim]
        # Empty condition for unconditioned image
        if image_guidance_scale != 1.0:
            uc_latents.append(condition.encode(self, empty=True))
        c_timesteps.append(torch.zeros([batch_size], device=device))
        c_projections.append(pooled_prompt_embeds)
        c_adapters.append(condition.adapter)
        # This complement_condition will be combined with the original image.
        # See the token integration of OminiControl2 [https://arxiv.org/abs/2503.08280]
        if condition.is_complement:
            complement_cond = tokens

    # Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    _, _, height, width = latents.shape
    image_seq_len = (height // self.transformer.config.patch_size) * (
        width // self.transformer.config.patch_size
    )
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.base_image_seq_len,
        self.scheduler.config.max_image_seq_len,
        self.scheduler.config.base_shift,
        self.scheduler.config.max_shift,
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler, num_inference_steps, device, timesteps, sigmas, mu=mu
    )
    num_warmup_steps = max(
        len(timesteps) - num_inference_steps * self.scheduler.order, 0
    )
    self._num_timesteps = len(timesteps)

    if kv_cache:
        attn_counter = 0
        for module in self.transformer.modules():
            if isinstance(module, Attention):
                setattr(module, "cache_idx", attn_counter)
                attn_counter += 1
        kv_cond = [[[], []] for _ in range(attn_counter)]
        kv_uncond = [[[], []] for _ in range(attn_counter)]

        def clear_cache():
            for storage in [kv_cond, kv_uncond]:
                for kesy, values in storage:
                    kesy.clear()
                    values.clear()

    branch_n = len(conditions) + 2
    group_mask = torch.ones([branch_n, branch_n], dtype=torch.bool)
    # Disable the attention cross different condition branches
    group_mask[2:, 2:] = torch.diag(torch.tensor([1] * len(conditions)))
    # Disable the attention from condition branches to image branch and text branch
    if kv_cache:
        group_mask[2:, :2] = False

    # Denoising loop
    # with self.progress_bar(total=num_inference_steps) as progress_bar:
    for i, t in enumerate(timesteps):
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = t.expand(latents.shape[0]).to(latents.dtype) / 1000

        if kv_cache:
            mode = "write" if i == 0 else "read"
            if mode == "write":
                clear_cache()
        use_cond = not (kv_cache) or mode == "write"
        # import ipdb
        # ipdb.set_trace()
        noise_pred = transformer_forward(
            self.transformer,
            image_features=[latents] + (c_latents if use_cond else []),
            text_features=[prompt_embeds],
            timesteps=[timestep, timestep] + (c_timesteps if use_cond else []),
            pooled_projections=[pooled_prompt_embeds] * 2
            + (c_projections if use_cond else []),
            return_dict=False,
            adapters=[main_adapter] * 2 + (c_adapters if use_cond else []),
            cache_mode=mode if kv_cache else None,
            cache_storage=kv_cond if kv_cache else None,
            to_cache=[False, False, *[True] * len(c_latents)],
            group_mask=group_mask,
            **transformer_kwargs,
        )[0]

        if image_guidance_scale != 1.0:
            unc_pred = transformer_forward(
                self.transformer,
                image_features=[latents] + (uc_latents if use_cond else []),
                text_features=[prompt_embeds],
                timesteps=[timestep, timestep] + (c_timesteps if use_cond else []),
                pooled_projections=[pooled_prompt_embeds] * 2
                + (c_projections if use_cond else []),
                return_dict=False,
                adapters=[main_adapter] * 2 + (c_adapters if use_cond else []),
                cache_mode=mode if kv_cache else None,
                cache_storage=kv_uncond if kv_cache else None,
                to_cache=[False, False, *[True] * len(c_latents)],
                **transformer_kwargs,
            )[0]

            noise_pred = unc_pred + image_guidance_scale * (noise_pred - unc_pred)

        # compute the previous noisy sample x_t -> x_t-1
        latents_dtype = latents.dtype
        latents = self.scheduler.step(noise_pred, t, latents)[0]

        if latents.dtype != latents_dtype:
            if torch.backends.mps.is_available():
                # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                latents = latents.to(latents_dtype)

        if callback_on_step_end is not None:
            callback_kwargs = {}
            for k in callback_on_step_end_tensor_inputs:
                callback_kwargs[k] = locals()[k]
            callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

            latents = callback_outputs.pop("latents", latents)
            prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

        # call the callback, if provided
            # if i == len(timesteps) - 1 or (
            #     (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            # ):
                # progress_bar.update()

    if latent_mask is not None:
        # Combine the generated latents and the complement condition
        assert complement_cond is not None
        comp_latent, comp_ids = complement_cond
        all_ids = torch.cat([latent_image_ids, comp_ids], dim=0)  # (Ta+Tc,3)
        shape = (all_ids.max(dim=0).values + 1).to(torch.long)  # (3,)
        H, W = shape[1].item(), shape[2].item()
        B, _, C = latents.shape
        # Create a empty canvas
        canvas = latents.new_zeros(B, H * W, C)  # (B,H*W,C)

        # Stash the latents and the complement condition
        def _stash(canvas, tokens, ids, H, W) -> None:
            B, T, C = tokens.shape
            ids = ids.to(torch.long)
            flat_idx = (ids[:, 1] * W + ids[:, 2]).to(torch.long)
            canvas.view(B, -1, C).index_copy_(1, flat_idx, tokens)

        _stash(canvas, latents, latent_image_ids, H, W)
        _stash(canvas, comp_latent, comp_ids, H, W)
        latents = canvas.view(B, H * W, C)

    if output_type == "latent":
        image = latents
    else:

        latents = (
            latents / self.vae.config.scaling_factor
        ) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    return StableDiffusion3PipelineOutput(images=image)
