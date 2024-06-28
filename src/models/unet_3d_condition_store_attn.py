import math

import torch
import torch.utils.checkpoint
from torch.nn import functional as F

from diffusers.models.unets.unet_3d_condition import UNet3DConditionModel
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from diffusers.utils import (
    USE_PEFT_BACKEND,
)

from .lavie.unet import LaVieModel


class AttnStore:
    def __init__(
        self,
    ):
        '''
        Initialize an empty AttentionStore
        '''
        self.attn_store = []
        self.is_able = False

    def __call__(self, attn):
        if self.is_able:
            self.attn_store.append(attn)

    def reset(self):
        for i in range(len(self.attn_store)):
            self.attn_store[i] = None
        self.attn_store.clear()

    def enable(self, is_able=True):
        self.is_able = is_able


class AttnProcessorStoreCrossAttn(AttnProcessor2_0):
    def __init__(
        self,
        attn_store: AttnStore,
        attn_res: int,
        phrase_indices: list,
        name: str
    ):
        super().__init__()
        self.attn_store = attn_store
        self.attn_res = attn_res
        self.phrase_indices = phrase_indices
        self.name = name

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale: float = 1.0,
    ):
        residual = hidden_states
        batch_size, vision_sequence_length, _ = hidden_states.shape
        is_cross = encoder_hidden_states is not None

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if is_cross and vision_sequence_length == self.attn_res and self.attn_store.is_able == True:
            # AttnProcessor + AttnStore
            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            # torch.set_printoptions(profile='full', linewidth=1000)
            # print(attention_probs[0].mean(0))
            # torch.set_printoptions(profile='default')
        
            self.attn_store(attention_probs[:, :, self.phrase_indices])

            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)
        else:
            # AttnProcessor2_0
            if attention_mask is not None:
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
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
    

def aggregate_attention(attn_store: AttnStore,
                        batch_size,
                        height,
                        width,
                        num_frames,) -> torch.Tensor:
    """ Aggregates the attention across the different layers and heads at the specified resolution. """
    out = []
    attention_maps = attn_store.attn_store

    for attn in attention_maps:
        if attn.shape[1] != height * width:
            scale_factor = int(math.sqrt(height * width // attn.shape[1]))
            attn = attn.reshape(-1, height // scale_factor, width // scale_factor, attn.shape[-1]).permute(0, 3, 1, 2) # b*f*num_heads, l, h, w
            attn = F.interpolate(attn, size=(height, width), mode='bilinear', align_corners=False)
            attn = attn.permute(0, 2, 3, 1)
        cross_maps = attn.reshape(batch_size, num_frames, -1, height, width, attn.shape[-1]).mean(2).mean(-1) # b, f, h, w
        out.append(cross_maps)
    out = torch.stack(out, dim=0).mean(0)
    attn_store.reset()
    return out


class UNet3DConditionAttnStoreModel(UNet3DConditionModel):
    def register_attn_store_processor(
        self,
        attn_store: AttnStore,
        attn_res: int,
        phrase_indices: list,
    ):
        attn_procs = {}
        for name in self.attn_processors.keys():
            attn_procs[name] = AttnProcessorStoreCrossAttn(attn_store=attn_store, attn_res=attn_res, phrase_indices=phrase_indices, name=name)
        self.set_attn_processor(attn_procs)


class LavieAttnStoreModel(LaVieModel):
    def register_attn_store_processor(
        self,
        attn_store: AttnStore,
        attn_res: int,
        phrase_indices: list,
    ):
        attn_procs = {}
        for name in self.attn_processors.keys():
            attn_procs[name] = AttnProcessorStoreCrossAttn(attn_store=attn_store, attn_res=attn_res, phrase_indices=phrase_indices, name=name)
        self.set_attn_processor(attn_procs)