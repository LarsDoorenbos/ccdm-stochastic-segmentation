
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
        cond_encoded_shape,
        backbone: str,
        backbone_params: Dict[str, Any],
        dataset_file: str,
        step_T_sample: str = None,
        feature_cond_encoder: dict = None
) -> DenoisingModel:

    img_shape, label_shape = input_shapes
    img_channels = img_shape[0]

    num_classes = label_shape[0]  
    diffusion = DiffusionModel(schedule, time_steps, num_classes, schedule_params=schedule_params)

    model: nn.Module

    if backbone == "unet_openai":
        model = create_unet_openai(
            image_size=min(img_shape[1], img_shape[2]),
            in_channels=num_classes + img_channels,
            out_channels=num_classes,
            num_res_blocks=2,
            cond_encoded_shape=cond_encoded_shape,
            feature_cond_encoder=feature_cond_encoder,
            **backbone_params
        )
    else:
        raise NotImplementedError(f"backbone {backbone}")

    num_of_parameters = sum(map(torch.numel, model.parameters()))
    LOGGER.info("%s trainable params: %d", backbone, num_of_parameters)

    return DenoisingModel(diffusion, model, dataset_file, step_T_sample)
