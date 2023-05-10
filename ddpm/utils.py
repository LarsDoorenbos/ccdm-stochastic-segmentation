import os
import shutil
import imageio
import pathlib
from typing import cast, Union, List, Union, Optional, List, Tuple, Text, BinaryIO
from types import FunctionType
from random import randint
import numpy as np

import torch
import torch as th
from torchvision.ops import focal_loss
from torchvision.transforms import ToPILImage
from PIL import Image
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader, Subset
import ignite.distributed as idist
from torchvision.utils import make_grid  # save_image

# Local imports
from .models import DenoisingModel
from .models.one_hot_categorical import OneHotCategoricalBCHW
from .models.condition_encoder import ConditionEncoder
from torch.nn.functional import one_hot

__all__ = [
    'ParallelType',
    'expanduservars',
    'archive_code',
    'WithStateDict',
    'worker_init_fn',
    '_onehot_to_color_image',
    '_flatten',
    '_loader_subset',
    'grid_of_predictions'
]

ParallelType = Union[nn.DataParallel, nn.parallel.DistributedDataParallel]

Model = Union[nn.Module, DenoisingModel,
              nn.parallel.DataParallel, nn.parallel.DistributedDataParallel]


def _flatten(m: Model) -> Union[DenoisingModel]:
    if isinstance(m, DenoisingModel):
        return m
    elif isinstance(m, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)):
        if isinstance(m.module, DenoisingModel):
            return cast(DenoisingModel, m.module)
        else:
            TypeError(
                f"type(m.module) should be one of (DenoisingModel, DataParallel, DistributedDataParallel) instead "
                f"got {type(m.module)}")
    else:
        raise TypeError(f"type(m) should be one of (DenoisingModel, DataParallel, DistributedDataParallel) instead got"
                        f" {type(m)}")


def _loader_subset(loader: DataLoader, num_images: int, randomize=False) -> DataLoader:
    dataset = loader.dataset
    lng = len(dataset)
    fixed_indices = range(0, lng - lng % num_images, lng // num_images)
    if randomize:
        overlap = True
        fixed_indices_set = set(fixed_indices)
        maxatt = 5
        cnt = 0
        while overlap and cnt < maxatt:
            indices = [randint(0, lng - 1) for _ in range(0, num_images)]
            overlap = len(set(indices).intersection(fixed_indices_set)) > 0
            cnt += 1
    else:
        indices = fixed_indices
    return DataLoader(
        Subset(dataset, indices),
        batch_size=loader.batch_size,
        shuffle=False
    )


# taken from torchvision utils
def save_image(tensor: Union[torch.Tensor, List[torch.Tensor]],
               fp: Union[Text, pathlib.Path, BinaryIO],
               format: Optional[str] = None,
               **kwargs) -> Image:
    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)
    return im


@torch.no_grad()
def grid_of_predictions(model: Model, cond_encoder: ConditionEncoder, loader: DataLoader, num_predictions: int,
                        cond_vis_fn: FunctionType, params: dict) -> Tensor:
    model.eval()
    cond_encoder.eval()
    diffusion_type = params["diffusion_type"] if "diffusion_type" in params else "categorical"

    labels_: List[Tensor] = []
    predictions_: List[Tensor] = []
    conditions_: List[Tensor] = []

    for batch in loader:
        images_b, labels_b = batch

        images_b = images_b.to(idist.device())
        if params["conditioning"] == 'concat_pixels_concat_features':
            condition_b_feats = cond_encoder(images_b)
            condition_b_enc = images_b
        else:
            condition_b_feats = None
            condition_b_enc = cond_encoder(images_b)

        labels_b = labels_b.to(idist.device())
        if diffusion_type == 'categorical':
            if len(batch) == 3:
                # mm cityscapes
                label_shape = (labels_b.shape[0], model.num_classes, *labels_b.shape[3:])
            else:
                # cityscapes and friends
                label_shape = (labels_b.shape[0], model.num_classes, *labels_b.shape[2:])

        elif diffusion_type == 'continuous_analog_bits':
            # validation loader applies one_hot so labels_b must be of
            if len(labels_b.shape) == 4:
                hw = labels_b.shape[2:]
            else:
                raise ValueError(f'invalid label shape [{labels_b.shape}]')
            label_shape = (labels_b.shape[0], model.bits, *hw) # in bit format
        else:
            raise ValueError(f'unknown diffusion_type given {diffusion_type}')

        predictions_b = []
        for _ in range(num_predictions):

            if diffusion_type == 'categorical':
                x = OneHotCategoricalBCHW(logits=torch.zeros(label_shape, device=labels_b.device)).sample()
                y = model(x, condition_b_enc, condition_features=condition_b_feats)["diffusion_out"]
                prediction = _onehot_to_color_image(y, params)
                predictions_b.append(prediction)
            else:
                raise NotImplementedError(f'diffusion_type {diffusion_type} not implemented')

        predictions_b = torch.stack(predictions_b, dim=1)
        labels_.append(_onehot_to_color_image(labels_b, params))
        predictions_.append(predictions_b)
        conditions_.append(cond_vis_fn(images_b))

    labels = torch.cat(labels_, dim=0).cpu()
    predictions = torch.cat(predictions_, dim=0).cpu()
    conditions = torch.cat(conditions_, dim=0).cpu()

    grid = torch.cat([
        conditions[:, None].expand((-1, -1, 3, -1, -1)),
        labels[:, None].expand((-1, -1, 3, -1, -1)),
        predictions.expand((-1, -1, 3, -1, -1))
    ],
        dim=1
    )
    return torch.reshape(grid, (-1,) + grid.shape[2:])


class WithStateDict(nn.Module):
    """Wrapper to provide a `state_dict` method to a single tensor."""

    def __init__(self, **tensors):
        super().__init__()
        for name, value in tensors.items():
            self.register_buffer(name, value)
        # self.tensor = nn.Parameter(tensor, requires_grad=False)


def scale_intensity(x):
    return 2.0 * x - 1.0


def expanduservars(path: str) -> str:
    return os.path.expanduser(os.path.expandvars(path))


def archive_code(path: str, params_filename="params.yml") -> None:
    shutil.copy(params_filename, path)
    # Copy the current code to the output folder.
    os.system(f"git ls-files -z | xargs -0 tar -czf {os.path.join(path, 'code.tar.gz')}")


def read_img_and_map(x):
    print(f"loading img {x[0]}")
    print(f"loading seg {x[1]}")
    return imageio.imread(x[0]), imageio.imread(x[1])


def save_x(xt: Tensor, output_path: str, timestep="t"):
    save_xt = True
    if save_xt:
        from ddpm.trainer import _onehot_to_color_image
        color_xt = _onehot_to_color_image(xt, {"dataset_file": "not_voc"})[0]
        color_xt = color_xt.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(color_xt)
        filename = output_path + f"/x_{timestep}.png"
        im.save(filename)
        print(f"saved {filename}")


def to_numpy(tensor):
    """Tensor to numpy, calls .cpu() if necessary"""
    with torch.no_grad():
        if tensor.device.type == 'cuda':
            tensor = tensor.cpu()
        return tensor.numpy()


def pil_from_bchw_tensor_label(x, save=False, name=''):
    # debugging function
    # from datasets.ade20k_config import decode_target_to_color
    from datasets.cityscapes_config import decode_target_to_color
    # x B,C,H,W
    assert len(x.shape) == 4
    b, c, h, w = x.shape

    if c == 1:  # integer valued labels
        x_int = x.squeeze(1).long()
    else:  # assume onehot otherwise
        x_int = torch.argmax(x, dim=1, keepdim=True).squeeze(1).long()

    # x_int is B1HW -> x_rgb is B,H,W,3
    x_rgb = decode_target_to_color(x_int)  # BCHW
    x_rgb_numpy = to_numpy(x_rgb).astype(np.uint8)

    # only show 1st image in the batch
    x_pil = Image.fromarray(x_rgb_numpy[0])

    if save:
        # x_pil.show()
        x_pil.save(name)
    return x_pil


def pil_from_bchw_tensor_image(x, save=False, name=''):
    # debugging function
    # from datasets.ade20k_config import decode_target_to_color
    from datasets.cityscapes_config import decode_target_to_color
    # x B,C,H,W
    assert len(x.shape) == 4
    b, c, h, w = x.shape
    x_pil = ToPILImage()(x[0])
    return x_pil


def worker_init_fn(_):
    # used for seeding dataloader workers in case mp_loader > 1
    # ensures numpy's randomness across epochs
    np.random.seed(torch.initial_seed() % 2**32)


def _onehot_to_color_image(source, params):
    if "cityscapes" in params["dataset_file"]:
        target = _onehot_to_cityscape_color_image(source)
    else:
        raise NotImplementedError(f"unknown color figure settings for dataset {params['dataset_file']}")
    return target


def _onehot_to_cityscape_color_image(arr: Tensor):
    from datasets.cityscapes_config import decode_target_to_color
    if len(arr.shape) == 4:
        B, C, H, W = arr.size()
        arr = arr.argmax(dim=1, keepdim=True)
        arr = decode_target_to_color(arr)
        arr = arr.permute(0, 4, 2, 3, 1)
        arr = torch.squeeze(arr)

    elif len(arr.shape) == 3:
        C, H, W = arr.size()
        B = 1
        arr = arr.argmax(dim=0, keepdim=True)
        arr = decode_target_to_color(arr)
        arr = arr.permute((3, 1, 2, 0))
        arr = torch.squeeze(arr)
        assert arr.shape == (C, H, W), f"{arr.shape} {C} {H} {W}"
    else:
        raise NotImplementedError

    if B == 1:
        arr = torch.unsqueeze(arr, dim=0)
    assert arr.shape == (B, 3, H, W), f"{arr.shape} {B} {C} {H} {W}"

    return arr / 255
