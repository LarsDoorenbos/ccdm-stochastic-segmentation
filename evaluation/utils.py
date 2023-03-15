import os

import torch
from torch import nn
from typing import cast, Union
from ddpm.models.diffusion_denoising import DenoisingModel
Model = Union[DenoisingModel, nn.parallel.DataParallel, nn.parallel.DistributedDataParallel]


def _flatten(m: Model) -> DenoisingModel:
    if isinstance(m, DenoisingModel):
        return m
    elif isinstance(m, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)):
        return cast(DenoisingModel, m.module)
    else:
        raise TypeError("type(m) should be one of (DenoisingModel, DataParallel, DistributedDataParallel)")


def reverse_mapping(mapping: dict):
    """ inverts class experiment mappings or id to name dicts """
    rev_mapping = dict()
    for key in mapping.keys():
        if key==255: continue
        vals = mapping[key]
        if isinstance(vals, list):
            for key_new in vals:
                rev_mapping[key_new] = [key]
        elif type(vals) == str:
            rev_mapping[vals] = key
    return rev_mapping


def to_numpy(tensor):
    """Tensor to numpy, calls .cpu() if necessary"""
    with torch.no_grad():
        if tensor.device.type == 'cuda':
            tensor = tensor.cpu()
        return tensor.numpy()


def create_new_directory(d):
    """create if it does not exist else do nothing and return -1"""
    _ = os.makedirs(d) if not(os.path.isdir(d)) else -1
    return _