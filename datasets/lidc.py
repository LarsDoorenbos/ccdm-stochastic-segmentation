
import os
import platform
import random
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
import h5py

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as tf
from torch.utils.data.dataset import Subset

if platform.node() == 'lars-HP-ENVY-Laptop-15-ep0xxx':
    FILE_PATH = "/home/lars/Outliers/data/data_lidc.hdf5"
elif platform.node() == "ue55290":
    FILE_PATH = "/home/lukas/datasets/data_lidc.hdf5"
else:
    FILE_PATH = "/storage/workspaces/artorg_aimi/ws_00000/lars/data_lidc.hdf5"

NUM_CLASSES = 2
RESOLUTION = 128
BACKGROUND_CLASS = None


def find_subset_for_id(ids_dict, id):
    for tt in ['test', 'train', 'val']:
        if id in ids_dict[tt]:
            return tt
    raise ValueError('id was not found in any of the train/test/val subsets.')


def process_data(file):
        max_bytes = 2**31 - 1
        hdf5_file = h5py.File('/storage/workspaces/artorg_aimi/ws_00000/lars/data_lidc.hdf5', "w")

        data = {}

        file_path = os.fsdecode(file)
        bytes_in = bytearray(0)
        input_size = os.path.getsize(file_path)
        with open(file_path, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        new_data = pickle.loads(bytes_in)
        data.update(new_data)

        series_uid = []

        for _, value in data.items():
            series_uid.append(value['series_uid'])

        unique_subjects = np.unique(series_uid)

        split_ids = {}
        train_and_val_ids, split_ids['test'] = train_test_split(unique_subjects, test_size=0.2)
        split_ids['train'], split_ids['val'] = train_test_split(train_and_val_ids, test_size=0.2)
        
        
        images = {}
        labels = {}
        uids = {}
        groups = {}

        for tt in ['train', 'test', 'val']:
            images[tt] = []
            labels[tt] = []
            uids[tt] = []
            groups[tt] = hdf5_file.create_group(tt)

        for key, value in data.items():

            s_id = value['series_uid']

            tt = find_subset_for_id(split_ids, s_id)

            images[tt].append(value['image'].astype(float)-0.5)

            lbl = np.asarray(value['masks'])  # this will be of shape 4 x 128 x 128

            labels[tt].append(lbl)
            uids[tt].append(hash(s_id))  # Checked manually that there are no collisions

        for tt in ['test', 'train', 'val']:

            groups[tt].create_dataset('uids', data=np.asarray(uids[tt], dtype=np.int))
            groups[tt].create_dataset('labels', data=np.asarray(labels[tt], dtype=np.uint8))
            groups[tt].create_dataset('images', data=np.asarray(images[tt], dtype=np.float))

        hdf5_file.close()


class LIDC_IDRI(Dataset):
    def __init__(self, dataset, transform=None):
        self.transform = transform
        self.dataset = dataset

    def __getitem__(self, index):
        image = np.expand_dims(self.dataset["images"][index], axis=0)
        label = self.dataset["labels"][index][random.randint(0,3)].astype(float)

        if self.transform is not None:
            image, label = self.transform(image, label)

        return image, label

    def __len__(self):
        return len(self.dataset["images"])


def one_hot_encoding(arr: np.ndarray) -> np.ndarray:
    res = np.zeros(arr.shape + (NUM_CLASSES,), dtype=np.float32)
    h, w = np.ix_(np.arange(arr.shape[0]), np.arange(arr.shape[1]))
    res[h, w, arr] = 1.0

    return res


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    if arr.ndim == 2:
        arr = arr[:, :, None]

    return torch.from_numpy(arr.transpose((2, 0, 1))).contiguous()


def transform(image, labels):
    labels = labels.astype(int)
    labels = one_hot_encoding(labels)

    image = tf.to_tensor(image.transpose((1, 2, 0))).float()
    labels = to_tensor(labels)

    if torch.rand(1) < 0.5:
        image = tf.hflip(image)
        labels = tf.hflip(labels)

    if torch.rand(1) < 0.5:
        image = tf.vflip(image)
        labels = tf.vflip(labels)

    rots = np.random.randint(0, 4)
    image = torch.rot90(image, rots, [1, 2])
    labels = torch.rot90(labels, rots, [1, 2])

    image = image * 2
    return image, labels


def training_dataset():
    dataset = LIDC_IDRI(h5py.File(FILE_PATH, 'r')['train'], transform)
    return dataset


def validation_dataset(max_size = 500):
    dataset = Test_LIDC(h5py.File(FILE_PATH, 'r')['val'], batch_transform)
    if max_size == None:
        return dataset
    dataset, _ = torch.utils.data.random_split(dataset, [max_size, len(dataset) - max_size], generator=torch.Generator().manual_seed(1))
    return dataset


def batch_transform(image, labels):
    image = tf.to_tensor(image.transpose((1, 2, 0))).float()
    image = image * 2

    for i in range(4):
        labels[str(i)] = labels[str(i)].astype(int)
        labels[str(i)] = one_hot_encoding(labels[str(i)])
        labels[str(i)] = to_tensor(labels[str(i)])

    labels = torch.cat((labels['0'][None], labels['1'][None], labels['2'][None], labels['3'][None]), dim=0)
    return image, labels


class Test_LIDC(Dataset):

    def __init__(self, dataset, transform):
        self.transform = transform
        self.dataset = dataset

    def __getitem__(self, index):
        image = np.expand_dims(self.dataset["images"][index], axis=0)

        # Select the four labels for this image
        labels = {}

        for i in range(4):
            labels[str(i)] = self.dataset["labels"][index][i]
        
        if self.transform is not None:
            image, labels = self.transform(image, labels)

        return image, labels, np.array([0.25, 0.25, 0.25, 0.25])

    def __len__(self):
        return len(self.dataset["images"])


def test_dataset(max_size=500, indices:list=None):
    dataset = Test_LIDC(h5py.File(FILE_PATH, 'r')['test'], batch_transform)

    if indices is not None:
        return Subset(dataset, indices)

    if max_size == None:
        return dataset

    return Subset(dataset, range(max_size))


def get_num_classes() -> int:
    return NUM_CLASSES


def get_ignore_class() -> int:
    return BACKGROUND_CLASS