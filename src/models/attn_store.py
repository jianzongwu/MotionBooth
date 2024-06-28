import torch
from torch.nn import functional as F

from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from diffusers.utils import (
    USE_PEFT_BACKEND,
)


class AttentionStore:
    def __init__(self, attention_res):
        '''
        Initialize an empty AttentionStore
        '''
        self.step_store = self.get_empty_store()
        self.attention_res = attention_res
        self.is_able = False

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": [],
                "transformer_in_self": []}

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if attn is None:
            return
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if is_cross and self.is_able:
            self.step_store[key].append(attn)

    def reset(self):
        self.step_store = self.get_empty_store()

    def enable(self, is_able=True):
        self.is_able = is_able


class AttnProcessorWithHook(AttnProcessor2_0):
    def __init__(
        self,
        attnstore,
        place_in_unet,
        attn_res,
        allow_low_res,
    ):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet
        self.attn_res = attn_res
        self.allow_low_res = allow_low_res

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
        batch_size, sequence_length, _ = hidden_states.shape
        if self.allow_low_res:
            is_store = sequence_length <= self.attn_res
        else:
            is_store = sequence_length == self.attn_res
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
        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        if self.attnstore is not None and is_store:
            query_batch_dim = attn.head_to_batch_dim(query)
            key_batch_dim = attn.head_to_batch_dim(key)
            value_batch_dim = attn.head_to_batch_dim(value)
            attention_probs = attn.get_attention_scores(query_batch_dim, key_batch_dim, attention_mask)

            self.attnstore(attention_probs, is_cross, self.place_in_unet)

            hidden_states = torch.bmm(attention_probs, value_batch_dim)
            hidden_states = attn.batch_to_head_dim(hidden_states)
        else:
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attention_mask is not None:
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

            attention_probs = None
            self.attnstore(attention_probs, is_cross, self.place_in_unet)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states