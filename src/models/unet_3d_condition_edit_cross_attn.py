import math

import torch
import torch.utils.checkpoint
import torch.nn.functional as F

from diffusers.models.unets.unet_3d_condition import UNet3DConditionModel
from diffusers.models.attention_processor import Attention, AttnProcessor
from diffusers.utils import (
    USE_PEFT_BACKEND,
)

from .lavie.unet import LaVieModel


class AttnEditProcessor:
    def __init__(
        self,
        num_frames,
        image_width,
        image_height,
    ):
        self.reset()
        self.enable(False)
        self.num_frames = num_frames
        self.image_width = image_width
        self.image_height = image_height

    def __call__(self, attention_scores):
        if not self.is_able or self.bbox is None:
            return attention_scores

        _, image_sequence_length, text_sequence_length = attention_scores.shape
        height = int(math.sqrt(image_sequence_length / (self.image_width / self.image_height)))
        width = image_sequence_length // height
        # f, head_dim, h, w, text_len
        attention_scores = attention_scores.reshape(self.num_frames, -1, height, width, text_sequence_length)

        box_range = [self.image_width, self.image_height, self.image_width, self.image_height]
        amp_mask = torch.zeros_like(attention_scores)
        amp_sample_mask = torch.zeros_like(attention_scores)
        bg_amp_mask = torch.ones_like(attention_scores[:, :, :, :, self.bg_phrase_indices])
        for obj_idx, indices in enumerate(self.phrase_indices):
            if indices is None:
                continue
            
            for frame_id, frame_box in enumerate(self.bbox):
                box = [max(round(amp_mask.shape[2 + (i + 1) % 2] * (b / r)), 0) for i, (b, r) in enumerate(zip(frame_box[obj_idx], box_range))]
                x1, y1, x2, y2 = box
                amp_mask[frame_id, :, y1:y2, x1:x2, indices] = 1.0
                bg_amp_mask[frame_id, :, y1:y2, x1:x2] = 0.0

        sup_mask = 1 - amp_mask
        if self.bg_phrase_indices is not None:
            sup_mask[:, :, :, :, self.bg_phrase_indices] = sup_mask[:, :, :, :, self.bg_phrase_indices] - bg_amp_mask
        sup_mask = sup_mask[:, :, :, :, 1:self.prompt_length - 1]

        attn_amp = amp_mask * (1 - amp_mask.sum((2,3), keepdim=True) / image_sequence_length)
        bg_attn_amp = bg_amp_mask * (1 - bg_amp_mask.sum((2,3), keepdim=True) / image_sequence_length)
        attn_sup = sup_mask * -1e10

        if self.is_amp:
            attention_scores = attention_scores + self.edit_scale * attn_amp
            if self.bg_phrase_indices is not None:
                attention_scores[:, :, :, :, self.bg_phrase_indices] = attention_scores[:, :, :, :, self.bg_phrase_indices] + self.edit_scale * bg_attn_amp    
        attention_scores[:, :, :, :, 1:self.prompt_length - 1] = attention_scores[:, :, :, :, 1:self.prompt_length - 1] + self.edit_scale * attn_sup

        attention_scores = attention_scores.reshape(-1, image_sequence_length, text_sequence_length)

        return attention_scores

    def reset(self):
        self.bbox = None
        self.phrase_indices = None
        self.bg_phrase_indices = None
        self.edit_scale = 0.0
        self.prompt_length = None
        self.is_amp = False

    def set_variables(self, bbox, phrase_indices, bg_phrase_indices, edit_scale, prompt_length, is_amp):
        self.bbox = bbox
        self.phrase_indices = phrase_indices
        self.bg_phrase_indices = bg_phrase_indices
        self.edit_scale = edit_scale
        self.prompt_length = prompt_length
        self.is_amp = is_amp

    def enable(self, is_able=True):
        self.is_able = is_able


class AttnProcessorEditCrossAttn(AttnProcessor):
    def __init__(
        self,
        attn_edit_processor: AttnEditProcessor,
    ):
        super().__init__()
        self.attn_edit_processor = attn_edit_processor

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

        query_batch_dim = attn.head_to_batch_dim(query)
        key_batch_dim = attn.head_to_batch_dim(key)
        value_batch_dim = attn.head_to_batch_dim(value)

        # attention_probs = attn.get_attention_scores(query_batch_dim, key_batch_dim, attention_mask)

        dtype = query_batch_dim.dtype
        if attn.upcast_attention:
            query_batch_dim = query_batch_dim.float()
            key_batch_dim = key_batch_dim.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query_batch_dim.shape[0], query_batch_dim.shape[1], key_batch_dim.shape[1], dtype=query_batch_dim.dtype, device=query_batch_dim.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query_batch_dim,
            key_batch_dim.transpose(-1, -2),
            beta=beta,
            alpha=attn.scale,
        )
        del baddbmm_input
        
        # edit cross attention
        if is_cross:
            attention_scores = self.attn_edit_processor(attention_scores)

        if attn.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        hidden_states = torch.bmm(attention_probs, value_batch_dim)
        hidden_states = attn.batch_to_head_dim(hidden_states)

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


class UNet3DConditionEditCrossAttnModel(UNet3DConditionModel):
    r"""
    A conditional 3D UNet model that can forward cache part only.

    This model inherits from [`UNet3DConditionModel`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).
    """
    def register_attn_edit_processor(
        self,
        attn_edit_processor: AttnEditProcessor,
    ):
        attn_procs = {}
        for name in self.attn_processors.keys():
            attn_procs[name] = AttnProcessorEditCrossAttn(attn_edit_processor=attn_edit_processor)
        self.set_attn_processor(attn_procs)


class LavieEditCrossAttnModel(LaVieModel):
    def register_attn_edit_processor(
        self,
        attn_edit_processor: AttnEditProcessor,
    ):
        attn_procs = {}
        for name in self.attn_processors.keys():
            attn_procs[name] = AttnProcessorEditCrossAttn(attn_edit_processor=attn_edit_processor)
        self.set_attn_processor(attn_procs)