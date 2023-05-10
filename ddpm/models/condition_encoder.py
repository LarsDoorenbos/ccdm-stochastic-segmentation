from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import ignite.distributed as idist
from datasets.pipelines import Denormalize
import logging
try:
    from .dino import ViTExtractor
except:
    from dino import ViTExtractor

import numpy as np

LOGGER = logging.getLogger(__name__)

__all__ = ["build_cond_encoder"]


class ConditionEncoder(nn.Module):
    def __init__(self):
        super().__init__()


class DummyEncoder(ConditionEncoder):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ResNetEncoder(ConditionEncoder):
    def __init__(self, train_encoder: bool, conditioning: str):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet34', pretrained=True)
        self.model.fc = nn.Identity()
        if not train_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
        self.conditioning = conditioning

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.fc(x)
        
        if self.conditioning == 'x-attention':
            x = rearrange(x, 'b f h w -> b (h w) f')
        elif self.conditioning == 'sum':
            x = F.adaptive_avg_pool2d(x, 1).squeeze()

        return x


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

    # def parameters(self):
    #     return self.extractor.model.parameters()
    #
    # def eval(self):
    #     self.extractor.model.eval()
    #
    # def train(self):
    #     self.extractor.model.train()

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, list]:
        f = self.extractor.extract_descriptors(x, self.layers, resize_shape=self.resize_shape)
        return f


def build_cond_encoder(params: dict):
    if params["cond_encoder"] == 'resnet' and params["conditioning"] != 'concat':
        cond_encoder = ResNetEncoder(params["train_encoder"], params["conditioning"]).to(idist.device())
        cond_vis_fn = lambda x: x * torch.tensor([0.229, 0.224, 0.225], device=x.device)[:, None, None] \
                                + torch.tensor([0.485, 0.456, 0.406], device=x.device)[:, None, None]

    elif "dino" in params["cond_encoder"]:
        cond_encoder = DinoViT(params["cond_encoder"],
                               params["train_encoder"],
                               params["conditioning"],
                               stride=params['cond_encoder_stride']).to(idist.device())
        cond_vis_fn = lambda x: x * torch.tensor([0.229, 0.224, 0.225], device=x.device)[:, None, None] \
                                + torch.tensor([0.485, 0.456, 0.406], device=x.device)[:, None, None]

    else:
        cond_encoder = DummyEncoder().to(idist.device())
        denorm = Denormalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        cond_vis_fn = lambda x: x / 2 + 0.5 if params["dataset_file"] in ['datasets.toy', 'datasets.toy_cont',
                                                                          'datasets.lidc'] else denorm(x)

    model_parameters = filter(lambda p: p.requires_grad, cond_encoder.parameters())
    num_parameters = sum([np.prod(p.size()) for p in model_parameters])
    LOGGER.info('Condition encoder trainable parameters: %d', num_parameters)

    if not isinstance(cond_encoder, DummyEncoder):
        if params["distributed"]:
            if params["train_encoder"]:  # add ddp to cond encoder only if training it
                local_rank = idist.get_local_rank()
                cond_encoder = nn.parallel.DistributedDataParallel(cond_encoder, device_ids=[local_rank])
        elif params["multigpu"]:
            cond_encoder = nn.DataParallel(cond_encoder)
    return cond_encoder, cond_vis_fn


if __name__ == '__main__':
    a = 1

    # encoder = ResNetEncoder(False, 'x-attention')

    p = {"cond_encoder": "dino_vits8", "dataset_file": "datasets.cityscapes"}

    encoder = ViTExtractor(p["cond_encoder"], stride=8, device="cuda")
    x_ = torch.randn(size=(2, 3, 128, 256))
    # encoder.model(x.float().cuda()) # (2, 384)
    # stride = 8
    # encoder.extract_descriptors(x.float().cuda()) # (2, 1, 512, 384) --> (2, 1, 16, 32, 384)
