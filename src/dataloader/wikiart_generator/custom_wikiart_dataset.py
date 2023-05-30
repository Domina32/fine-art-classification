from pathlib import Path
from typing import Optional

from torchvision.transforms import functional as fn

from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from src.dataloader.dataloader import CustomDataset
from src.helpers.image_utils import resize_img

local_FILE_LENGTH_MAP_JSON_PATH = Path(__file__).parent


class CustomWikiartDataset(CustomDataset):
    def __init__(
        self,
        chosen_label="genre",
        chunk_size=1,
    ):
        super().__init__(chosen_label=chosen_label, chunk_size=chunk_size)

        dataset = load_dataset("huggan/wikiart", cache_dir="./data/wikiart/dataset")

        if isinstance(dataset, DatasetDict):
            dataset = dataset["train"]
            column_names_to_remove = [
                column_name
                for column_name in dataset.column_names
                if column_name != chosen_label and column_name != "image"
            ]
            dataset = dataset.remove_columns(column_names_to_remove)
            self.__dataset = dataset.with_format("torch")
        else:
            raise ValueError(f"Wrong type of dataset. Expected {DatasetDict}, got {type(dataset)}")

    def __len__(self):
        return len(self.__dataset)

    def __getitem__(self, key):
        image = self.__dataset[key]["image"]

        return (
            fn.convert_image_dtype(resize_img(image.permute(2, 0, 1))),
            self.__dataset[key][self.chosen_label].item(),
        )

    def __iter__(self):
        return self.generator()

    def generator(self):
        for item in self.__dataset:
            if isinstance(item, dict):
                image = item["image"]
                yield (
                    fn.convert_image_dtype(resize_img(image.permute(2, 0, 1))),
                    item[self.chosen_label].item(),
                )
            else:
                raise ValueError(f"Expected item in dataset to have type {dict}, found {type(item)}")

    def slice(self, start: int = 0, stop: Optional[int] = None, step: int = 1):
        self.__dataset = self.__dataset.select(range(start, stop or self.__length, step))

        return self


# import ast
# import json
# import os
# import torch
# import torchvision
# import numpy as np
# import pandas as pd

# def init(
#     self,
#     use_huggingface=False,
#     wikiart_data_path="wikiart",
#     file_length_map_json_path=local_FILE_LENGTH_MAP_JSON_PATH,
#     chosen_label="genre",
#     chunk_size=1,
# ):
#     temp_ = 0
#     with open(os.path.join(file_length_map_json_path, "file_length_map.json"), "r") as file_id_map:
#         map_dict = json.load(file_id_map)
#         for file_length in map_dict.values():
#             temp_ += file_length

#     self.file_length_map_json_path = file_length_map_json_path  # type: ignore
#     self.data_path = PurePath(self.data_path, wikiart_data_path)
#     self.length = temp_


# def csv_getitem(self, raw_row_id):
#     with open(os.path.join(self.file_length_map_json_path, "file_length_map.json"), "r") as file_id_map:
#         map_dict = json.load(file_id_map)

#         file_number = int(raw_row_id)
#         file_name = ""
#         for file_name, file_length in map_dict.items():
#             if file_number - int(file_length - 1) < 0:
#                 break
#             else:
#                 file_number -= int(file_length - 1)

#         file_length = int(map_dict.get(file_name, -1))
#         assert file_length > 0, f"No key called {file_name}"

#         row_id_in_file = raw_row_id % file_length + 1 if raw_row_id >= file_length else raw_row_id

#     row = pd.read_csv(f"{self.data_path}/csv/" + file_name, skiprows=range(1, row_id_in_file), nrows=1)

#     return (
#         resize_img(
#             torchvision.io.decode_image(
#                 torch.tensor(np.frombuffer(ast.literal_eval(row["image"].iloc[0]).get("bytes"), dtype=np.uint8))
#             )
#         ),
#         row[self.chosen_label].iloc[0],
#     )


# def csv_generator(self):
#     with open(os.path.join(self.file_length_map_json_path, "file_length_map.json"), "r") as file_id_map:
#         map_dict = json.load(file_id_map)
#         for file_name in map_dict:
#             for chunk in pd.read_csv(f"{self.data_path}/csv/" + file_name, chunksize=self.chunk_size):
#                 image_arrays, labels = [], []
#                 image_arrays = torch.tensor(
#                     chunk["image"]
#                     .map(
#                         lambda img: resize_img(
#                             torchvision.io.decode_image(
#                                 torch.tensor(np.frombuffer(ast.literal_eval(img).get("bytes"), dtype=np.uint8))
#                             )
#                         )
#                     )
#                     .values
#                 )
#                 labels = chunk[self.chosen_label].values  # .decode('utf-8')

#                 if self.chunk_size == 1:
#                     for row in range(self.chunk_size):
#                         yield (image_arrays[row], labels[row])

#                 yield (
#                     image_arrays,
#                     np.asarray(labels),
#                 )
