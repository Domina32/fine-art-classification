import os
from typing import Literal, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
from numpy import uint8
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader

from src.dataloader.dataloader import split_dataset
from src.dataloader.wga_generator.custom_wga_dataset import CustomWgaDataset
from src.dataloader.wikiart_generator.custom_wikiart_dataset import CustomWikiartDataset
from src.helpers.image_utils import label_mapping


def paintings_label_mapping():
    path = "./data/wga/"
    paintings = pd.read_csv(os.path.join(path, "paintings.csv"))
    paintings["encoded_TYPE"] = paintings["TYPE"].map(label_mapping)
    print("Paintings label mapping done")

    temp = paintings[["new_URL", "encoded_TYPE"]].copy()
    temp.to_csv(os.path.join(path, "temp.csv"), index=False)


def get_dataloader(
    data: Union[Literal["wga"], Literal["wikiart"]],
    test_split,
    batch_size,
    shuffle,
    random_seed,
    check=False,
    image_save_name="./example.png",
    num_workers=0,
    prefetch_factor: Optional[int] = None,
    pin_memory: bool = False,
    slice: Optional[Union[tuple[int, int], tuple[int, int, int]]] = None,
) -> tuple[DataLoader[tuple[Union[NDArray[uint8], Tensor]]], DataLoader[tuple[NDArray[uint8]]]]:
    if data == "wga":
        dataset = CustomWgaDataset(chunk_size=1) if not slice else CustomWgaDataset(chunk_size=1).slice(*slice)
    elif data == "wikiart":
        dataset = CustomWikiartDataset() if not slice else CustomWikiartDataset().slice(*slice)
    else:
        raise ValueError(f"Unknown argument {data}")

    train_loader, test_loader = split_dataset(
        dataset,
        test_split,
        batch_size,
        shuffle,
        random_seed,
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
