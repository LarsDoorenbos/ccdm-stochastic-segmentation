
import logging
from typing import Dict, List, Tuple, Any, Union

import torch
from torch import nn
from .diffusion_denoising import DiffusionModel, DenoisingModel
from .unet_openai import create_unet_openai

LOGGER = logging.getLogger(__name__)


def build_model(
        time_steps: int,
        schedule: str,
        schedule_params: Union[dict, None],
        input_shapes: List[Tuple[int, int, int]],
        guidance_scale: float,
        guidance_scale_weighting: str,
        guidance_loss_fn: str,
        label_smoothing: float,
        conditioning: str,
        cond_encoded_shape,
        backbone: str,
        backbone_params: Dict[str, Any],
        dataset_file: str,
        step_T_sample: str = None,
        diffusion_type: str = 'categorical',
        bits: int = None,
        analog_bits_scale: float = 1.0,
        # feature_cond_target_output_stride: int = 8,
        # feature_cond_target_module_index: int = 11,
        params=None
) -> DenoisingModel:

    img_shape, label_shape = input_shapes
    img_channels = img_shape[0]

    # diffusion type: affects input/output shape
    if diffusion_type == 'categorical':
        num_classes = label_shape[0]  # can be accessed from this cause label is in one_hot encoding
        LOGGER.info(f"Using diffusion model [{diffusion_type}] with num_classes [{num_classes}]")
        diffusion = DiffusionModel(schedule, time_steps, num_classes, schedule_params=schedule_params)
        input_channels = num_classes
        output_channels = num_classes
    else:
        raise NotImplementedError(f'unknown diffusion {diffusion_type} in params.yml')

    if conditioning in ['concat', 'concat_pixels_concat_features', 'concat_pixels_attend_features']:
        input_channels += img_channels  # cases where image is used as input conditioning

    backbone_params_implicit = {}
    backbone_params_implicit.update({'in_channels': input_channels,
                                     'out_channels': output_channels,
                                     'conditioning': conditioning,
                                     'cond_encoded_shape': cond_encoded_shape,
                                     'num_res_blocks': 2})

    LOGGER.info(f"Implicit backbone params: {backbone_params_implicit}")
    LOGGER.info(f"Explicit backbone params: {backbone_params}")
    backbone_params.update(backbone_params_implicit)

    model: nn.Module
    if backbone == "unet_openai":
        model = create_unet_openai(image_size=min(img_shape[1], img_shape[2]), **backbone_params, params=params)
    else:
        raise NotImplementedError(f"backbone {backbone}")

    num_of_parameters = sum(map(torch.numel, model.parameters()))
    LOGGER.info("%s trainable params: %d", backbone, num_of_parameters)

    if params is not None:
        # for logging only
        params['num_params'] = num_of_parameters
        params['unet_openai'] = backbone_params

    ret = DenoisingModel(diffusion, model, guidance_scale, guidance_scale_weighting, guidance_loss_fn,
                         dataset_file, label_smoothing, step_T_sample)
    return ret
