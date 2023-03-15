
import logging
from typing import Union

import numpy as np

import torch
import torch.nn as nn
import ignite.distributed as idist

from datasets.pipelines.transforms import Denormalize

try:
    from .dino import ViTExtractor
except:
    from dino import ViTExtractor

LOGGER = logging.getLogger(__name__)


class ConditionEncoder(nn.Module):
    def __init__(self):
        super().__init__()


class DinoViT(ConditionEncoder):
    def __init__(self, name: str,
                 train_encoder: bool,
                 conditioning: str,
                 stride: int = 8,
                 resize_shape: Union[tuple, None] = None,
                 layers: Union[list, int] = 11):
        super().__init__()
        self.extractor = ViTExtractor(name, stride)
        if not train_encoder:
            for param in self.parameters():
                param.requires_grad = False

        self.stride = stride
        self.conditioning = conditioning
        self.layers = layers
        self.resize_shape = resize_shape

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, list]:
        f = self.extractor.extract_descriptors(x, self.layers, resize_shape=self.resize_shape)
        return f


def create_cond_fis_fn_default(params):
    denorm = Denormalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    cond_vis_fn = lambda x: x / 2 + 0.5 if params["dataset_file"] in ['datasets.lidc', 'datasets.lidc_orig'] else denorm(x)
    return cond_vis_fn


def _build_feature_cond_encoder(params: dict):
    fce_params = params["feature_cond_encoder"]
    if 'dino' in fce_params['type']:
        feature_cond_encoder = DinoViT(fce_params["model"],
                                       fce_params["train"],
                                       fce_params["conditioning"],
                                       stride=fce_params['output_stride']).to(idist.device())
        model_parameters = filter(lambda p: p.requires_grad, feature_cond_encoder.parameters())
        num_parameters = sum([np.prod(p.size()) for p in model_parameters])
        LOGGER.info(f"Feature Condition encoder {fce_params} parameters: {num_parameters}")
        LOGGER.info(f"Feature Condition encoder parameters {num_parameters}")

        cond_vis_fn = lambda x: x * torch.tensor([0.229, 0.224, 0.225], device=x.device)[:, None, None] \
                                + torch.tensor([0.485, 0.456, 0.406], device=x.device)[:, None, None]
        
        if params["distributed"]:
            if fce_params["train"]:  # add ddp to cond encoder only if training it
                local_rank = idist.get_local_rank()
                feature_cond_encoder = nn.parallel.DistributedDataParallel(feature_cond_encoder, device_ids=[local_rank])
        elif params["multigpu"]:
            feature_cond_encoder = nn.DataParallel(feature_cond_encoder)
    else:
        feature_cond_encoder = None
        cond_vis_fn = create_cond_fis_fn_default(params)
        LOGGER.info(f"No Feature Condition encoder in use.")

    return feature_cond_encoder, cond_vis_fn