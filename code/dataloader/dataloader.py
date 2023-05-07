from typing import Optional

import numpy as np
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


BATCH_SIZE = 16
TEST_SPLIT = 0.1
SHUFFLE = True
RANDOM_SEED = 32


def split_dataset(
    dataset: CustomDataset,
    test_split=TEST_SPLIT,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    random_seed=RANDOM_SEED,
    num_workers=0,
    prefetch_factor: Optional[int] = None,
    pin_memory: bool = False,
):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(TEST_SPLIT * dataset_size))
    if SHUFFLE:
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=test_sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
    )

    return (train_loader, test_loader)
