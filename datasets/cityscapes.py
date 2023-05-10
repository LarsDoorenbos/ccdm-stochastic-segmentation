
import platform
import os

import numpy as np

import torch
from torchvision import transforms, datasets

import ddpm

from datasets.cityscapes_config import encode_target, classes
from typing import Union

BASE_PATH = os.path.expandvars("C:\\Users\\thopis\\Documents\\datasets\\CITYSCAPES")
# BASE_PATH = os.path.expandvars("${TMPDIR}/cityscapes/")

NUM_CLASSES = 20
BACKGROUND_CLASS = 19

NORMALIZER = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
COLOR_JITTER = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)


def get_weights(class_weights):
    if class_weights == 'uniform':
        # return torch.as_tensor([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        # cf. cityscape_config.py
        return torch.as_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    # DeepLab used this class weight for cityscapes
    elif class_weights == 'weighted':
        return torch.as_tensor(
            [0, 0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
             1.0865, 1.1529, 1.0507])


def labels_to_categories(arr: np.ndarray) -> np.ndarray:
    return encode_target(arr)


def training_dataset(transforms_dict_train: Union[dict, None] = None):
    dataset = datasets.Cityscapes(root=BASE_PATH, split="train", mode="fine", target_type="semantic")
    # noinspection PyTypeChecker
    dataset = ddpm.TransformedImgLblDataset(dataset,
                                            transforms_dict_train,
                                            num_classes=get_num_classes(),
                                            label_mapping_func=labels_to_categories
                                            )

    return dataset


def validation_dataset(max_size: int = 64, transforms_dict_val: Union[dict, None] = None):
    dataset = datasets.Cityscapes(root=BASE_PATH, split="val", mode="fine", target_type="semantic")
    # noinspection PyTypeChecker
    dataset = ddpm.TransformedImgLblDataset(dataset,
                                            transforms_dict_val,
                                            num_classes=get_num_classes(),
                                            label_mapping_func=labels_to_categories
                                            )

    if max_size:
        dataset, _ = torch.utils.data.random_split(dataset, [max_size, len(dataset) - max_size], generator=torch.Generator().manual_seed(1))
    return dataset


def test_dataset(max_size: int = 128):
    return validation_dataset(max_size)


def get_num_classes() -> int:
    return NUM_CLASSES


def get_ignore_class() -> int:
    return BACKGROUND_CLASS


def train_ids_to_class_names() -> dict:
    train_ids_to_classes = {}
    for i in range(get_num_classes()):
        for c in classes:
            if c.train_id == i:
                train_ids_to_classes[i] = c.name
    return train_ids_to_classes
