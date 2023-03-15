import pathlib
import types
from typing import (Generic, TypeVar, Callable, Sequence,
                    Pattern, Union, List, Dict, Any, cast, Optional)
from inspect import isfunction
import h5py
import imageio
import torch.nn.functional

from torchvision.transforms import Compose, ToPILImage, ToTensor
from torch.utils.data import Dataset

__all__ = [
    "EmptyDataset",
    "H5Dataset",
    "FileListDataset",
    "FileListReDataset",
    "TransformedDataset",
    "TransformedImgLblDataset"
]

Tin = TypeVar('Tin')
Tout = TypeVar('Tout')


class EmptyDataset(Dataset):

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError('`EmptyDataset` is empty')


class H5Dataset(Dataset):

    def __init__(self, h5file: str, dataset_key: str):
        self.h5file = h5py.File(h5file, 'r')
        self.dataset = self.h5file[dataset_key]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class FileListDataset(Dataset, Generic[Tin, Tout]):

    def __init__(self,
                 file_list: Sequence[Tin],
                 loader: Callable[[Tin], Tout] = imageio.imread
                 ) -> None:
        self.loader = loader
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tout:
        return self.loader(self.file_list[idx])


class FileListReDataset(FileListDataset, Generic[Tin, Tout]):

    def __init__(self,
                 file_list: Sequence[Tin],
                 regexp: Pattern,
                 labels: Union[List[str], Dict[str, Any]],
                 loader: Callable[[Tin], Tout] = imageio.imread
                 ) -> None:

        super().__init__(file_list, self._loader)
        self.regexp = regexp
        self.base_loader = loader

        if isinstance(labels, list):
            labels = {lbl: i for i, lbl in enumerate(labels)}
        self.labels = cast(Dict[str, Any], labels)

    def _loader(self, filename: Tin):

        x = self.base_loader(filename)

        match = self.regexp.match(filename)
        if match is None:
            raise ValueError(
                "could not find a match with file name `{}`".format(filename)
            )
        grp = match.group(1)
        label = self.labels[grp]

        return x, label


class TransformedDataset(Dataset, Generic[Tin, Tout]):

    def __init__(self,
                 source_dataset: Dataset,
                 transform_func: Callable[..., Tout]
                 ) -> None:
        self.source_dataset = source_dataset
        self.transform_func = transform_func

    def __len__(self):
        return len(self.source_dataset)

    def __getitem__(self, idx: int) -> Tout:

        value = self.source_dataset[idx]

        if isinstance(value, tuple):
            return self.transform_func(*value)

        return self.transform_func(value)


class TransformedImgLblDataset(Dataset):
    def __init__(self,
                 source_dataset: Dataset,
                 transforms_dict: Dict,
                 num_classes: int,
                 label_mapping_func: Optional[types.FunctionType] = None,
                 return_metadata=False):
        """
        Modified TransformedDataset class that uses three sets of transformation functions:
            "common" : applied to both img and lbl
            "img" : applied to img only
            "lbl" : applied to lbl only
        transformations in each set are composed using torchvision.transforms.Compose
        these are user-defined in params.yml and datasets.pipelines.transforms.build_transforms
        :param source_dataset: torch Dataset (ex. torchvision.
        :param transforms_dict: dict that must have 3 keys "common", "img", "lbl", each with value a list
         (see build_transforms for more details)
        :param label_mapping_func: a function that maps annotated labels to labels used for training
        :param num_classes:
        """

        assert(('common' in transforms_dict) and ('img' in transforms_dict) and ('lbl' in transforms_dict))
        super(Dataset, self).__init__()

        self.source_dataset = source_dataset
        self.transform_dict = transforms_dict
        self.common_transforms = Compose(transforms_dict['common'])
        self.img_transforms = Compose(transforms_dict['img'])
        self.label_mapping_func = label_mapping_func
        self.num_classes = num_classes

        if isfunction(self.label_mapping_func):
            pos = len(transforms_dict['lbl']) - 1
            transforms_dict['lbl'].insert(pos, self.label_mapping_func)
        else:
            assert(self.label_mapping_func is None), f'invalid label_mapping_func: {self.label_mapping_func}'
        self.lbl_transforms = Compose(transforms_dict['lbl'])

        # only important in evaluation
        self.return_metadata = return_metadata

        self.apply_one_hot = True  # add this to the pipelines
        self.debug = False

    def __len__(self):
        return len(self.source_dataset)

    def __getitem__(self, idx: int) -> Tout:

        image, target = self.source_dataset[idx]

        metadata = {'index': idx}
        image, target, metadata = self.common_transforms((image, target, metadata))
        img_tensor = self.img_transforms(image)
        lbl_tensor = self.lbl_transforms(target).squeeze()

        if self.debug:
            import numpy as np
            ToPILImage()(img_tensor).show()
            ToPILImage()(lbl_tensor).show()
            print(f'\nafter aug index, : {np.unique(lbl_tensor)}  lbl {lbl_tensor.shape} image {img_tensor.shape}')
            return img_tensor, lbl_tensor, metadata

        if self.apply_one_hot:
            lbl_tensor = torch.nn.functional.one_hot(lbl_tensor.long(), self.num_classes)
            lbl_tensor = torch.permute(lbl_tensor, (2, 0, 1))

        if self.return_metadata:
            # write custom collate to allow batched passing of filenames from dataloader,
            # otherwise such metadata can only be passed out with batch_size=1
            # if hasattr(self.source_dataset, "images"):
            #     filepath = self.source_dataset.images[idx]
            #     filename = pathlib.Path(filepath).stem
            # elif hasattr(self.source_dataset.dataset, "images"):
            #     filepath = self.source_dataset.dataset.images[idx]
            #     filename = pathlib.Path(filepath).stem
            # else:
            #     raise ValueError()
            original_lbl_tensor = metadata_transform(metadata, self.label_mapping_func)[0]
            return img_tensor, lbl_tensor, original_lbl_tensor

        return img_tensor, lbl_tensor


def metadata_transform(metadata: dict, label_mapping_function: types.FunctionType) -> dict:
    # fix to map id->train_id in original labels, stored in the metadata dict
    if 'original_labels' in metadata:
        original_lbl_tensor = label_mapping_function(metadata['original_labels']).squeeze()
    return ToTensor()(original_lbl_tensor.astype('int32'))

