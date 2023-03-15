
from .unet import UNetModel


def create_unet_openai(
    image_size,
    base_channels,
    in_channels,
    out_channels,
    num_res_blocks,
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
    feature_cond_encoder=None
):

    if channel_mult is None:
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4) 
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    
    return UNetModel(
        in_channels=in_channels,
        model_channels=base_channels,
        out_channels=out_channels,
        num_res_blocks=num_res_blocks,
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
        feature_cond_encoder=feature_cond_encoder
    )
