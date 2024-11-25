import math
from typing import Any, Dict, Optional, Union, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from dataclasses import dataclass

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import is_torch_version, logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import TimestepEmbedding, Timesteps, get_3d_sincos_pos_embed, get_timestep_embedding
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNorm  # , CogVideoXLayerNormZero
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU, FP32SiLU, SwiGLU
from einops import rearrange
from diffusers.utils import BaseOutput
import numpy as np

class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        kv_heads: Optional[int] = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        qk_norm: Optional[str] = None,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        processor: Optional["AttnProcessor"] = None,
        out_dim: int = None,
        context_pre_only=None,
        pre_only=False,
    ):
        super().__init__()

        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.fused_projections = False
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only

        # we make use of this private variable to know whether this class is loaded
        # with an deprecated state dict so that we can convert it on the fly
        self._from_deprecated_attn_block = _from_deprecated_attn_block

        self.scale_qk = scale_qk
        self.scale = torch.sqrt(torch.tensor(dim_head)) / 8

        self.heads = out_dim // dim_head if out_dim is not None else heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads

        self.added_kv_proj_dim = added_kv_proj_dim
        self.only_cross_attention = only_cross_attention

        if self.added_kv_proj_dim is None and self.only_cross_attention:
            raise ValueError(
                "`only_cross_attention` can only be set to True if `added_kv_proj_dim` is not None. Make sure to set either `only_cross_attention=False` or define `added_kv_proj_dim`."
            )

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True)
        else:
            self.group_norm = None
        
        self.spatial_norm = None

        if qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        elif qk_norm == "layer_norm":
            self.norm_q = nn.LayerNorm(dim_head, eps=eps)
            self.norm_k = nn.LayerNorm(dim_head, eps=eps)

        self.norm_cross = None
        
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)

        if not self.only_cross_attention:
            # only relevant for the `AddedKVProcessor` classes
            self.to_k = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
            self.to_v = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        else:
            self.to_k = None
            self.to_v = None

        self.added_proj_bias = added_proj_bias
        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_kv_dim, bias=added_proj_bias)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_kv_dim, bias=added_proj_bias)
            if self.context_pre_only is not None:
                self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)

        if not self.pre_only:
            self.to_out = nn.ModuleList([])
            self.to_out.append(nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
            self.to_out.append(nn.Dropout(dropout))

        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_add_out = nn.Linear(self.inner_dim, self.out_dim, bias=out_bias)

        self.norm_added_q = None
        self.norm_added_k = None

    def forward(
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        grow_masks=None,
        rope_pe: Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.size()
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.size() if encoder_hidden_states is None else encoder_hidden_states.size()
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.size(-1))

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        # if grow_masks is not None and "d_model" in grow_masks:
        #     query = query * grow_masks["d_model"]
        #     key = key * grow_masks["d_model"]
        #     value = value * grow_masks["d_model"]

        inner_dim = key.size(-1)
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if grow_masks is not None and "d_head" in grow_masks:
            query = query * grow_masks["d_head"]
            key = key * grow_masks["d_head"]
            value = value * grow_masks["d_head"]

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if rope_pe is not None:
            query, key = apply_rope(query, key, rope_pe)

        # Hack - look out for future updates so we don't have to do this
        # query = (query * torch.sqrt(torch.tensor(query.shape[-1]))) / 8
        query = query * attn.scale

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False, #scale=1 / 8
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)

        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta ** scale)
    out = pos.unsqueeze(-1) * omega  # Broadcasting multiplication
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "... n d (i j) -> ... n d i j", i=2, j=2)
    return out.float()

def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.size()[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.size()[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.size()).type_as(xq), xk_out.reshape(*xk.size()).type_as(xk)


class CogXFeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias)
        elif activation_fn == "swiglu":
            act_fn = SwiGLU(dim, inner_dim, bias=bias)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for idx, module in enumerate(self.net):
            hidden_states = module(hidden_states)

        return hidden_states


class CogXLayerNormZero(nn.Module):
    def __init__(
        self,
        conditioning_dim: int,
        embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_dim, 6 * embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, temb: torch.Tensor, grow_masks=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate, enc_shift, enc_scale, enc_gate = self.linear(self.silu(temb)).chunk(6, dim=1)
    
        hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        encoder_hidden_states = (
            self.norm(encoder_hidden_states) * (1 + enc_scale)[:, None, :] + enc_shift[:, None, :]
        )
        return hidden_states, encoder_hidden_states, gate[:, None, :], enc_gate[:, None, :]


@maybe_allow_in_graph
class CogVideoXBlock(nn.Module):
    r"""
    Transformer block used in [CogVideoX](https://github.com/THUDM/CogVideo) model.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()

        # 1. Self Attention
        self.norm1 = CogXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        # processor = CogXAttnProcessor2_0()
        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            # processor=processor,
        )

        # 2. Feed Forward
        self.norm2 = CogXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.ff = CogXFeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rope_pe: torch.Tensor = None,
        grow_masks=None,
    ) -> torch.Tensor:
        # all_hidden_states = []
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb, grow_masks=grow_masks
        )

        # all_hidden_states.append((norm_hidden_states, norm_encoder_hidden_states))

        # attention
        text_length = norm_encoder_hidden_states.size(1)

        # CogVideoX uses concatenated text + video embeddings with self-attention instead of using
        # them in cross-attention individually
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        attn_output = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            grow_masks=grow_masks,
            rope_pe=rope_pe,
        )

        hidden_states = hidden_states + gate_msa * attn_output[:, text_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_output[:, :text_length]

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb, grow_masks=grow_masks
        )

        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states, grow_masks=grow_masks)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_length]

        return hidden_states, encoder_hidden_states


class CogVideoXLayerNormZeroWithZeroInit(nn.Module):
    r"""
    NOTE: CogVideoXLayerNormZero with ZERO initilized gates.
    """

    def __init__(
        self,
        conditioning_dim: int,
        embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_dim, 4 * embedding_dim, bias=bias)
        self.linear_part_2 = nn.Linear(conditioning_dim, 2 * embedding_dim, bias=bias)

        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

        # Zero init the gate values such that we preserve prev functionality.
        nn.init.zeros_(self.linear_part_2.weight)
        if self.linear_part_2.bias is not None:
            nn.init.zeros_(self.linear_part_2.bias)

    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        activated_embd = self.silu(temb)
        shift, scale, enc_scale, enc_shift = self.linear(activated_embd).chunk(4, dim=1)
        gate, enc_gate = self.linear_part_2(self.silu(temb)).chunk(2, dim=1)

        hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        encoder_hidden_states = self.norm(encoder_hidden_states) * (1 + enc_scale)[:, None, :] + enc_shift[:, None, :]

        return hidden_states, encoder_hidden_states, gate[:, None, :], enc_gate[:, None, :]



class CogVideoXBlockSoftMoe(nn.Module):
    r"""
    Modified Transformer block used in CogVideoX model. NOTE two properties stands out:
    1. it allows for the use of SoftMoe FF layers (when num_experts > 1) and normal FF layers (num_experts <= 1)
    2. it uses a zero initialized gating to make sure added modules of this kind do not alter original behavior.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        num_experts: int = 16,
        slots_per_expert: int = 1,
    ):
        super().__init__()

        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZeroWithZeroInit(
            time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True
        )

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
        )

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZeroWithZeroInit(
            time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True
        )

        self.ff = CogXFeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rope_pe: torch.Tensor = None,
        grow_masks=None,
    ) -> torch.Tensor:
        # all_hidden_states = []
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        # attention
        text_length = norm_encoder_hidden_states.size(1)

        # CogVideoX uses concatenated text + video embeddings with self-attention instead of using
        # them in cross-attention individually
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        attn_output = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            grow_masks=grow_masks,
            rope_pe=rope_pe,
        )

        hidden_states = hidden_states + gate_msa * attn_output[:, text_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_output[:, :text_length]

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states, grow_masks=grow_masks)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_length]

        return hidden_states, encoder_hidden_states


class CogXAdaLayerNorm(nn.Module):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`, *optional*): The size of the embeddings dictionary.
        output_dim (`int`, *optional*):
        norm_elementwise_affine (`bool`, defaults to `False):
        norm_eps (`bool`, defaults to `False`):
        chunk_dim (`int`, defaults to `0`):
    """

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: Optional[int] = None,
        output_dim: Optional[int] = None,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()

        self.chunk_dim = chunk_dim
        output_dim = output_dim or embedding_dim * 2

        if num_embeddings is not None:
            self.emb = nn.Embedding(num_embeddings, embedding_dim)
        else:
            self.emb = None

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        grow_masks=None,
    ) -> torch.Tensor:
        if self.emb is not None:
            temb = self.emb(timestep)

        temb = self.linear(self.silu(temb))

        if self.chunk_dim == 1:
            # This is a bit weird why we have the order of "shift, scale" here and "scale, shift" in the
            # other if-branch. This branch is specific to CogVideoX for now.
            shift, scale = temb.chunk(2, dim=1)
            shift = shift[:, None, :]
            scale = scale[:, None, :]
        else:
            scale, shift = temb.chunk(2, dim=0)

        x = self.norm(x) * (1 + scale) + shift
        return x


class ZeroCondInitTimestepEmbedding(TimestepEmbedding):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
    ):
        super().__init__(
            in_channels=in_channels,
            time_embed_dim=time_embed_dim,
            act_fn=act_fn,
            out_dim=out_dim,
            post_act_fn=post_act_fn,
            cond_proj_dim=cond_proj_dim,
        )

        if self.cond_proj is not None:
            nn.init.zeros_(self.cond_proj.weight)



class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.size()[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


class CogVideoXPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        embed_dim: int = 1920,
        text_embed_dim: int = 4096,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size

        # self.proj = nn.Linear(
        #     in_channels, embed_dim, bias=bias
        # )

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
        )
        self.text_proj = nn.Linear(text_embed_dim, embed_dim)

    # def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor, grow_masks=None):
    #     r"""
    #     Args:
    #         text_embeds (`torch.Tensor`):
    #             Input text embeddings. Expected shape: (batch_size, seq_length, embedding_dim).
    #         image_embeds (`torch.Tensor`):
    #             Input image embeddings. Expected shape: (batch_size, num_frames, channels, height, width).
    #     """
    #     text_embeds = self.text_proj(text_embeds)

    #     batch, N, channels = image_embeds.shape
    #     image_embeds = self.proj(image_embeds)

    #     embeds = torch.cat(
    #         [text_embeds, image_embeds], dim=1
    #     ).contiguous()  # [batch, seq_length + num_frames x height x width, channels]
    #     return embeds
    def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor, grow_masks=None):
        r"""
        Args:
            text_embeds (`torch.Tensor`):
                Input text embeddings. Expected shape: (batch_size, seq_length, embedding_dim).
            image_embeds (`torch.Tensor`):
                Input image embeddings. Expected shape: (batch_size, num_frames, channels, height, width).
        """
        text_embeds = self.text_proj(text_embeds)

        batch, num_frames, channels, height, width = image_embeds.size()
        image_embeds = image_embeds.reshape(-1, channels, height, width)
        image_embeds = self.proj(image_embeds)
        image_embeds = image_embeds.view(batch, num_frames, *image_embeds.size()[1:])
        image_embeds = image_embeds.flatten(3).transpose(2, 3)  # [batch, num_frames, height x width, channels]
        image_embeds = image_embeds.flatten(1, 2)  # [batch, num_frames x height x width, channels]

        embeds = torch.cat(
            [text_embeds, image_embeds], dim=1
        ).contiguous()  # [batch, seq_length + num_frames x height x width, channels]
        return embeds


class QuantisedCogVideoXTransformer3DModel(ModelMixin, ConfigMixin):
    """
    A Transformer model for video-like data in [CogVideoX](https://github.com/THUDM/CogVideo).

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input.
        out_channels (`int`, *optional*):
            The number of channels in the output.
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlocks` attention should contain a bias parameter.
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            This is fixed during training since it is used to learn a number of position embeddings.
        patch_size (`int`, *optional*):
            The size of the patches to use in the patch embedding layer.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to use in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*):
            The number of diffusion steps used during training. Pass if at least one of the norm_layers is
            `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings that are
            added to the hidden states. During inference, you can denoise for up to but not more steps than
            `num_embeds_ada_norm`.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The type of normalization to use. Options are `"layer_norm"` or `"ada_layer_norm"`.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether or not to use elementwise affine in normalization layers.
        norm_eps (`float`, *optional*, defaults to 1e-5): The epsilon value to use in normalization layers.
        caption_channels (`int`, *optional*):
            The number of channels in the caption embeddings.
        video_length (`int`, *optional*):
            The number of frames in the video-like data.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: Optional[int] = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        first_frame_conditioned: bool = False,
        zero_init_expand=False,
        ues_torch_compile: bool = True,
        original_innder_dim: int = 1920,
        use_rope: bool = False,
        additional_layer_locs: list = [],
        num_experts: int = 16,
        use_cond_timestep_proj: bool = False,
        slots_per_expert: int = 1,
    ):
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads
        inner_dim = num_attention_heads * attention_head_dim
        self.in_channels = in_channels
        self.out_channels = out_channels

        post_patch_height = sample_height // patch_size
        post_patch_width = sample_width // patch_size
        post_time_compression_frames = (sample_frames - 1) // temporal_compression_ratio + 1
        self.num_patches = post_patch_height * post_patch_width * post_time_compression_frames

        # 1. Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(patch_size, in_channels, inner_dim, text_embed_dim, bias=True)
        self.embedding_dropout = nn.Dropout(dropout)

        txt_embedding = torch.zeros(1, max_text_seq_length, inner_dim, requires_grad=False)
        self.register_buffer("txt_embedding", txt_embedding, persistent=False)

        # Rope
        theta = 10000
        axes_dim = [32, 48, 48]
        self.pe_embedder = EmbedND(dim=attention_head_dim, theta=theta, axes_dim=axes_dim)

        # 3. Time embeddings
        self.time_proj = Timesteps(original_innder_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = ZeroCondInitTimestepEmbedding(
            original_innder_dim,
            time_embed_dim,
            timestep_activation_fn,
            cond_proj_dim=original_innder_dim if use_cond_timestep_proj else None,
        )

        total_layers = num_layers + len(additional_layer_locs)
        adjusted_additional_layer_locs = [loc + i + 1 for i, loc in enumerate(additional_layer_locs)]

        # Initialize the transformer_blocks
        self.transformer_blocks = nn.ModuleList()
        original_block_idx = 0
        additional_block_idx = 0

        for layer_idx in range(total_layers):
            if layer_idx in adjusted_additional_layer_locs:
                # Insert the additional layer
                self.transformer_blocks.append(
                    CogVideoXBlockSoftMoe(
                        dim=inner_dim,
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=attention_head_dim,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        activation_fn=activation_fn,
                        attention_bias=attention_bias,
                        norm_elementwise_affine=norm_elementwise_affine,
                        norm_eps=norm_eps,
                        num_experts=num_experts,
                        slots_per_expert=slots_per_expert,
                    )
                )
                additional_block_idx += 1
            else:
                # Insert the original layer
                self.transformer_blocks.append(
                    CogVideoXBlock(
                        dim=inner_dim,
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=attention_head_dim,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        activation_fn=activation_fn,
                        attention_bias=attention_bias,
                        norm_elementwise_affine=norm_elementwise_affine,
                        norm_eps=norm_eps,
                    )
                )
                original_block_idx += 1


        self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)

        # 5. Output blocks
        self.norm_out = CogXAdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels)

        self.gradient_checkpointing = True
        self.grow_masks = {}
        self.ues_torch_compile = ues_torch_compile

        self.zero_init_expand = zero_init_expand
        self.iter = 0
        self.training = False

        # self.initialize_weights()

    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        # timestep_cond: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):

        # batch_size, N, in_channels, = hidden_states.shape
        batch_size, num_frames, in_channels, height, width = hidden_states.size()

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)
        # t_emb_cond = self.time_proj(timestep_cond).to(dtype=hidden_states.dtype) if timestep_cond is not None else None

        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, None)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        rope_pe = self.pe_embedder(ids)

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states, grow_masks=self.grow_masks)

        # hidden_states = self.embedding_dropout(hidden_states)

        encoder_hidden_states = hidden_states[:, : self.config.max_text_seq_length]
        hidden_states = hidden_states[:, self.config.max_text_seq_length :]

        # 5. Transformer blocks
        for block in self.transformer_blocks:
            
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=emb,
                rope_pe=rope_pe,
                grow_masks=self.grow_masks,
            )

        hidden_states = self.norm_final(hidden_states)

        # # 6. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb, grow_masks=self.grow_masks)
        hidden_states = self.proj_out(hidden_states)


        p = self.config.patch_size
        output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, self.out_channels, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

        if self.training:
            self.iter += 1

        if not return_dict:
            return (output,)
        return output