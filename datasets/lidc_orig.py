
import platform
import glob
import os
import random 

import numpy as np
from numpy import random
from torch.utils.data.dataset import Dataset

import torch
import torchvision.transforms.functional as tf
import imageio

if platform.node() == 'lars-HP-ENVY-Laptop-15-ep0xxx':
    BASE_PATH = "/home/lars/Outliers/data/LIDC"
else:
    BASE_PATH = "/storage/workspaces/artorg_aimi/ws_00000/lars/LIDC"

NUM_CLASSES = 2
BACKGROUND_CLASS = None
RESOLUTION = 128


class LIDC_dataset(Dataset):
    def __init__(self, file_list, seg_file_list, transform):
        self.transform = transform
        self.file_list = file_list
        self.seg_file_list = seg_file_list

    def __getitem__(self, index):
        image = imageio.imread(self.file_list[index])
        label = imageio.imread(self.seg_file_list[index, random.randint(0,3)])
        
        image, label = self.transform(image, label)
        return image, label

    def __len__(self):
        return len(self.file_list)
    

def one_hot_encoding(arr: np.ndarray) -> np.ndarray:
    res = np.zeros(arr.shape + (NUM_CLASSES,), dtype=np.float32)
    h, w = np.ix_(np.arange(arr.shape[0]), np.arange(arr.shape[1]))
    res[h, w, arr] = 1.0

    return res


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    if arr.ndim == 2:
        arr = arr[:, :, None]

    return torch.from_numpy(arr.transpose((2, 0, 1))).contiguous()


def training_transform(image, labels):
    labels = labels / 255
    labels = one_hot_encoding(labels.astype(int))

    image = tf.to_tensor(image)
    labels = tf.to_tensor(labels)

    image = tf.center_crop(image, (RESOLUTION, RESOLUTION))
    labels = tf.center_crop(labels, (RESOLUTION, RESOLUTION))
    
    image = image * 2 - 1

    if torch.rand(1) < 0.5:
        image = tf.hflip(image)
        labels = tf.hflip(labels)

    if torch.rand(1) < 0.5:
        image = tf.vflip(image)
        labels = tf.vflip(labels)

    rots = np.random.randint(0, 4)
    image = torch.rot90(image, rots, [1, 2])
    labels = torch.rot90(labels, rots, [1, 2])

    return image, labels


def training_dataset():
    seg_file_list = np.array(sorted(glob.glob(os.path.join(BASE_PATH, "lidc_crops_train/train/gt/*/*.png"))))
    file_list = sorted(glob.glob(os.path.join(BASE_PATH, "lidc_crops_train/train/images/*/*.png")))

    seg_file_list = seg_file_list.reshape((len(file_list), 4), order='C')
    
    dataset = LIDC_dataset(file_list, seg_file_list, training_transform)

    return dataset


def validation_dataset(max_size: int):
    seg_file_list = np.array(sorted(glob.glob(os.path.join(BASE_PATH, "lidc_crops_val/val/gt/*/*.png"))))
    file_list = sorted(glob.glob(os.path.join(BASE_PATH, "lidc_crops_val/val/images/*/*.png")))
    
    seg_file_list = seg_file_list.reshape((len(file_list), 4), order='C')

    dataset = LIDC_test_dataset(file_list, seg_file_list, batch_transform)

    if max_size:
        dataset, _ = torch.utils.data.random_split(dataset, [max_size, len(dataset) - max_size], generator=torch.Generator().manual_seed(1))
    return dataset


class LIDC_test_dataset(Dataset):
    def __init__(self, file_list, seg_file_list, transform):
        self.transform = transform
        self.file_list = file_list
        self.seg_file_list = seg_file_list

    def __getitem__(self, index):
        image = imageio.imread(self.file_list[index])

        labels = {}

        for i in range(4):
            labels[str(i)] = imageio.imread(self.seg_file_list[index, i])

        image, labels = self.transform(image, labels)

        labels = torch.cat((labels['0'][None], labels['1'][None], labels['2'][None], labels['3'][None]), dim=0)
        return image, labels, np.array([0.25, 0.25, 0.25, 0.25])

    def __len__(self):
        return len(self.file_list)


def batch_transform(image, labels):
    for i in range(4):
        labels[str(i)] = labels[str(i)] / 255
        labels[str(i)] = one_hot_encoding(labels[str(i)].astype(int))
        labels[str(i)] = to_tensor(labels[str(i)])
        labels[str(i)] = tf.center_crop(labels[str(i)], (RESOLUTION, RESOLUTION))

    image = tf.to_tensor(image)
    image = tf.center_crop(image, (RESOLUTION, RESOLUTION))

    image = image * 2 - 1
    return image, labels


def test_dataset(max_size):
    seg_file_list = np.array(sorted(glob.glob(os.path.join(BASE_PATH, "lidc_crops_test/test/gt/*/*.png"))))
    file_list = sorted(glob.glob(os.path.join(BASE_PATH, "lidc_crops_test/test/images/*/*.png")))

    seg_file_list = seg_file_list.reshape((len(file_list), 4), order='C')

    dataset = LIDC_test_dataset(file_list, seg_file_list, batch_transform)

    if max_size:
        dataset, _ = torch.utils.data.random_split(dataset, [max_size, len(dataset) - max_size], generator=torch.Generator().manual_seed(1))
        
    return dataset


def get_num_classes() -> int:
    return NUM_CLASSES


def get_ignore_class() -> int:
    return BACKGROUND_CLASS

