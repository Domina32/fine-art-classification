from typing import Optional

import numpy as np
import numpy.random
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

DEFAULT_IN_SHAPE = (3, 300, 300)


class CustomDataset(Dataset):
    def __init__(
        self,
        chosen_label="",
        chunk_size=1,
    ):
        self.chosen_label = chosen_label
        self.chunk_size = chunk_size
        self.in_shape = DEFAULT_IN_SHAPE
        self.__length = 0

    def __len__(self):
        return self.__length

    def __iter__(self):
        return self.generator()

    def generator(self):
        raise NotImplementedError()


def split_dataset(
    dataset: CustomDataset,
    test_split: int,
    batch_size: int,
    shuffle: bool,
    random_seed,
    num_workers=0,
    prefetch_factor: Optional[int] = None,
    pin_memory: bool = False,
):
    """
    Split dataset into two non-intersecting parts.
    """
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    if shuffle:
        random = numpy.random.default_rng(random_seed)
        random.shuffle(indices)

    train_indices, test_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
    )

    return (train_loader, test_loader)


def preprocess_dataset():
    """
    Apply preprocessing for resnet and densenet built-in models.
    From torch docs:
    "All pre-trained models expect input images normalized in the same way,
    i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
    where H and W are expected to be at least 224.
    The images have to be loaded in to a range of [0, 1] and then normalized
    using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]."
    """
    return
