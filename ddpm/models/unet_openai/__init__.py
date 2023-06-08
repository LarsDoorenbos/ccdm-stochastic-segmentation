from .unet import UNetModel
from typing import Union
import logging
LOGGER = logging.getLogger(__name__)


def create_unet_openai(
    image_size,
    base_channels,
    in_channels,
    out_channels,
    num_res_blocks,
    conditioning,
    cond_encoded_shape,
    channel_mult=None,
    use_checkpoint=False,
    attention_resolutions=[32, 16, 8],
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    softmax_output=True,
    ce_head=False,
    use_stem=False,
    skip_op='concat',
    num_res_blocks_per_dblock=None,
    feature_cond_channels: Union[tuple, int] = 384,
    feature_cond_target_output_stride=8,  # set for dino-vits8, stride 8, unet with max stride 32
    feature_cond_target_module_index=11,  # set for dino-vits8, stride 8, unet with max stride 32
    num_res_blocks_dec=-1,
    is_lightweight=False,
    is_with_swin=True,
    use_multiscale_predictions=False,
    multiscale_prediction_resolutions=(1, 2, 4, 8, 16, 32),
    params=None
):

    if channel_mult is None:
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)  # 30M params 256x512 - Bsize 16
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
        if params is not None:
            LOGGER.info(f"defaulting channel_mult to {channel_mult} for input of min size {image_size}")
            params['unet_openai']['channel_mult'] = channel_mult

    return UNetModel(
        in_channels=in_channels,
        model_channels=base_channels,
        out_channels=out_channels,
        num_res_blocks=num_res_blocks,
        num_res_blocks_dec=num_res_blocks,
        conditioning=conditioning,
        cond_encoded_shape=cond_encoded_shape,
        attention_resolutions=attention_resolutions,
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=None,
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
        softmax_output=softmax_output,
        ce_head=ce_head,
        feature_cond_target_output_stride=feature_cond_target_output_stride,
        feature_cond_target_module_index=feature_cond_target_module_index
    )
