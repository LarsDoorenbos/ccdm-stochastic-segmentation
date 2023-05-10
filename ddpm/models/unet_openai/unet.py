
from abc import abstractmethod
from typing import Union
import math

import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from .attention import SpatialTransformer


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None, feature_condition=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            elif isinstance(layer, ResBlock):
                x = layer(x, feature_condition)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        num_res_blocks_dec,
        conditioning,
        cond_encoded_shape,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        softmax_output=True,
        ce_head=False,
        use_stem=False,
        feature_cond_target_output_stride: int = 8,  # if feature condition is being used default to inserting it at stride 8
        feature_cond_target_module_index: Union[list, int] = 11
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.num_res_blocks_dec = num_res_blocks_dec
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.conditioning = conditioning
        self.cond_encoded_shape = cond_encoded_shape
        self.sofmtax_output = softmax_output
        self.use_ce_head = ce_head

        # used only if conditioning == 'concat_pixels_concat_features'
        # must specify output_stride (equivalent to 'level') at which features are provided
        # must specify index (in a given level) of resblock where features are provided alongside its input
        self.using_feature_cond = False
        self.feature_cond_target_output_stride = feature_cond_target_output_stride
        self.feature_cond_target_module_index = feature_cond_target_module_index
        self.feature_cond_channels = 384  # todo unhardcode, this is specific to dino-vits8

        time_embed_dim = model_channels * 4
        if use_stem:
            self.stem = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), padding=1, stride=2)
        else:
            self.stem = None

        if self.conditioning == 'sum':
            self.cond_embedder = nn.Linear(cond_encoded_shape[-1], time_embed_dim)
        elif self.conditioning == 'concat_linproj':
            self.linear_projection = nn.Conv2d(in_channels=in_channels + 3, out_channels=out_channels, kernel_size=(1, 1))
        elif self.conditioning == 'concat_pixels_concat_features':
            self.using_feature_cond = True
            self.using_pixel_cond = True

        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )

        # used to collect module details
        module_index = 0
        self.level_to_output_stride = dict()
        module_index = self.collect_module_info(module_index, 1, in_channels, ch, module_type='Conv2d')

        self._feature_size = ch
        input_block_chans = [ch]
        output_stride = 2 if use_stem else 1

        self.max_output_stride = 2**(len(channel_mult)-1)

        for level, mult in enumerate(channel_mult):

            for r in range(num_res_blocks):
                if all([output_stride == self.feature_cond_target_output_stride,
                        module_index == self.feature_cond_target_module_index,
                        self.using_feature_cond]):
                    ch_in = self.feature_cond_channels + ch
                else:
                    ch_in = ch

                module_index = self.collect_module_info(module_index, output_stride,
                                                        ch_in, int(mult * model_channels),
                                                        module_type='ResBlock')

                layers = [
                    ResBlock(
                        ch_in,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm
                    )]

                ch = int(mult * model_channels)  # next resblocks input channels
                if output_stride in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if use_new_attention_order:
                        dim_head = ch // num_heads

                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        ) if not self.conditioning == 'x-attention'
                        else SpatialTransformer(ch, num_heads, dim_head, depth=1, context_dim=cond_encoded_shape[2])
                    )

                    module_type = 'AttentionBlock' if not self.conditioning == 'x-attention' else 'SpatialTransformer'
                    module_index = self.collect_module_info(module_index-1, output_stride,
                                                            ch, (num_heads, num_head_channels),
                                                            module_type, no_increment=False)

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                module_type = 'ResBlockDown' if resblock_updown else 'Down'
                module_index = self.collect_module_info(module_index, output_stride,
                                                        ch, out_ch,
                                                        module_type)

                ch = out_ch
                input_block_chans.append(ch)
                output_stride *= 2
                self._feature_size += ch

        #  middle
        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if use_new_attention_order:
            dim_head = ch // num_heads

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ) if not self.conditioning == 'x-attention' else SpatialTransformer(
                ch, num_heads, dim_head, depth=1, context_dim=cond_encoded_shape[2]
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        _ = self.collect_module_info(0, output_stride, ch, ch, 'ResBlock_Attention_ResBlock', 'middle')

        self._feature_size += ch
        #  decoder
        module_index = 0
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks_dec + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]

                module_index = self.collect_module_info(module_index, output_stride,
                                                        ch + ich, int(mult * model_channels),
                                                        module_type='ResBlock', stage='dec')

                ch = int(model_channels * mult)
                if output_stride in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if use_new_attention_order:
                        dim_head = ch // num_heads

                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        ) if not self.conditioning == 'x-attention' else SpatialTransformer(
                            ch, num_heads, dim_head, depth=1, context_dim=cond_encoded_shape[2]
                        )
                    )

                    module_type = 'AttentionBlock' if not self.conditioning == 'x-attention' else 'SpatialTransformer'
                    module_index = self.collect_module_info(module_index-1, output_stride,
                                                            ch, (num_heads, num_head_channels),
                                                            module_type, stage='dec', no_increment=False)

                if level and i == num_res_blocks_dec:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )

                    module_type = 'ResBlockUp' if resblock_updown else 'Up'
                    module_index = self.collect_module_info(module_index-1, output_stride,
                                                            ch, out_ch,
                                                            module_type, stage='dec', no_increment=False)

                    output_stride //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        if self.sofmtax_output:
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
                nn.Softmax(dim=1)
            )
        else:
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1))
            )

        module_type = 'Head' if self.sofmtax_output else 'Head_Softmax'
        module_index = self.collect_module_info(module_index, output_stride,
                                                input_ch, out_channels,
                                                module_type, stage='out')

        # option for a parallel ce head
        if self.use_ce_head:
            print('adding an extra ce_head (distinct from diffusion_head) to Unet')
            print('No softmax is applied to this, only logits are returned ')
            self.out_ce = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                zero_module(conv_nd(dims, input_ch, out_channels-1, 3, padding=1))
            )
            module_type = 'Head_ce'
            module_index = self.collect_module_info(module_index, output_stride,
                                                    input_ch, out_channels,
                                                    module_type, stage='out')

            # out_channels-1 is valid only if dataset has ignore class fixme
        else:
            self.out_ce = None

    def collect_module_info(self,
                            module_index: int,
                            output_stride: int,
                            in_channels: int,
                            out_channels: Union[tuple, int],
                            module_type: str,
                            stage='enc',
                            no_increment=False):

        if stage not in self.level_to_output_stride:
            self.level_to_output_stride[stage] = dict()

        # assert module_index not in self.level_to_output_stride[stage]
        if module_index in self.level_to_output_stride[stage]:
            module_type = [self.level_to_output_stride[stage][module_index]['type'], module_type]
            in_channels = [self.level_to_output_stride[stage][module_index]['in_channels'], in_channels]
            out_channels = [self.level_to_output_stride[stage][module_index]['out_channels'], out_channels]
            self.level_to_output_stride[stage][module_index].update({'in_channels': in_channels,
                                                                     'out_channels': out_channels,
                                                                     'type': module_type})
        else:
            self.level_to_output_stride[stage][module_index] = dict()  # no repetitions of module index
        self.level_to_output_stride[stage][module_index] = {'os': output_stride,
                                                            'in_channels': in_channels,
                                                            'out_channels': out_channels,
                                                            'type': module_type}
        if no_increment:
            return module_index
        else:
            module_index += 1
            return module_index

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, condition, timesteps, feature_condition: Union[None, torch.Tensor] = None, y=None,
                get_multiscale_predictions=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param feature_condition:
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        b,c,H,W = x.shape

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.conditioning == 'sum':
            condition = self.cond_embedder(condition)
            emb = emb + condition
            context = None
        elif self.conditioning == 'x-attention':
            context = condition
        elif self.conditioning == 'concat':
            x = th.cat([x, condition], dim=1)
            context = None
        elif self.conditioning == 'concat_linproj':
            x = self.linear_projection(th.cat([x, condition], dim=1))
            context = None

        elif self.conditioning == 'concat_pixels_concat_features':
            x = th.cat([x, condition], dim=1)
            context = None

        if self.stem is not None:
            x = self.stem(x)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for level, module in enumerate(self.input_blocks):
            if self.using_feature_cond and \
                    (self.level_to_output_stride['enc'][level]['os'] == self.feature_cond_target_output_stride) and\
                    (self.feature_cond_target_module_index == level):
                h = th.cat([h, feature_condition], dim=1)

            # inshape = h.shape
            h = module(h, emb, context)
            # outshape = h.shape
            # print(f'Module: {module.__class__}, inshape: {inshape} outshape: {outshape}')
            hs.append(h)

        # bottleneck
        h = self.middle_block(h, emb, context)
        # print(f"---bottlenech {h.shape}---")
        for level, module in enumerate(self.output_blocks):
            # inshape = h.shape
            # v = hs.pop()
            h = th.cat([h, hs.pop()], dim=1)
            # except:
            #     a = 1
            h = module(h, emb, context)
            # outshape = h.shape
            # print(f'Module: {module.__class__}, inshape: {inshape} outshape: {outshape}')

        h = h.type(x.dtype)

        # output handling
        ret = {"diffusion_out": None, "logits": None}
        diffusion_out = self.out(h)

        if self.stem is not None:
            diffusion_out = torch.nn.functional.interpolate(diffusion_out, (H,W), mode='bilinear')

        ret["diffusion_out"] = diffusion_out

        if self.out_ce is not None:
            ce_out = self.out_ce(h)
            if self.stem is not None:
                ce_out = torch.nn.functional.interpolate(diffusion_out, (H,W), mode='bilinear')
            ret["logits"] = ce_out
        return ret