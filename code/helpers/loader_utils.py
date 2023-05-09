from code.dataloader.dataloader import split_dataset
from code.dataloader.wga_generator.CustomWgaDataset import CustomWgaDataset
from code.dataloader.wikiart_generator.CustomWikiartDataset import CustomWikiartDataset
from typing import Optional, Union

import matplotlib.pyplot as plt
from numpy import uint8
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader

BATCH_SIZE = 16


def get_dataloader(
    data="wga",
    check=False,
    image_save_name="./example.png",
    batch_size=BATCH_SIZE,
    num_workers=0,
    prefetch_factor: Optional[int] = None,
    pin_memory: bool = False,
    slice: Optional[Union[tuple[int, int], tuple[int, int, int]]] = None,
) -> tuple[
    DataLoader[tuple[Union[NDArray[uint8], Tensor]]], DataLoader[tuple[NDArray[uint8]]]
]:
    if data == "wga":
        dataset = (
            CustomWgaDataset(chunk_size=1)
            if not slice
            else CustomWgaDataset(chunk_size=1).slice(*slice)
        )
    elif data == "wikiart":
        dataset = (
            CustomWikiartDataset()
            if not slice
            else CustomWikiartDataset().slice(*slice)
        )
    else:
        raise ValueError(f"Unknown argument {data}")

    train_loader, test_loader = split_dataset(
        dataset,
        batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
    )

    if check:
        train_features, train_labels = next(iter(train_loader))
        print(f"Feature batch shape: {train_features.size()}")
        print(f"Labels batch shape: {train_labels.size()}")
        print(f"Label: {train_labels[0]}")

        if data == "wga":
            img = train_features[0]
        elif data == "wikiart":
            img = train_features[0].squeeze()
        else:
            raise ValueError(f"Unknown argument {data}")

        if image_save_name:
            plt.imshow(img, cmap="gray")
            plt.savefig(image_save_name)

    return train_loader, test_loader
