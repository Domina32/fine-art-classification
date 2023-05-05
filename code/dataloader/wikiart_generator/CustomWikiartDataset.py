import os
import json
import ast

import pandas as pd
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset

from helpers.image_utils import resize_img, change_channels


local_WIKIART_DATA_PATH = "../../../data/wikiart/"
local_FILE_LENGTH_MAP_JSON_PATH = "."


class CustomWikiartDataset(Dataset):
    def __init__(
        self,
        WIKIART_DATA_PATH=local_WIKIART_DATA_PATH,
        FILE_LENGTH_MAP_JSON_PATH=local_FILE_LENGTH_MAP_JSON_PATH,
        chosen_label="genre",
        chunk_size=1,
    ):
        temp_ = 0
        with open(os.path.join(FILE_LENGTH_MAP_JSON_PATH, "file_length_map.json"), "r") as file_id_map:
            map_dict = json.load(file_id_map)
            for file_length in map_dict.values():
                temp_ += file_length

        self.file_length_map_json_path = FILE_LENGTH_MAP_JSON_PATH
        self.wikiart_data_path = WIKIART_DATA_PATH
        self.length_of_all_files = temp_
        self.chosen_label = chosen_label
        self.chunk_size = chunk_size
        self.in_shape = (300, 300, 3)

    def __len__(self):
        return self.length_of_all_files

    def __getitem__(self, raw_row_id):
        with open(os.path.join(self.file_length_map_json_path, "file_length_map.json"), "r") as file_id_map:
            map_dict = json.load(file_id_map)

            file_number = int(raw_row_id)
            file_name = ""
            for file_name, file_length in map_dict.items():
                if file_number - int(file_length - 1) < 0:
                    break
                else:
                    file_number -= int(file_length - 1)

            file_length = int(map_dict.get(file_name, -1))
            assert file_length > 0, f"No key called {file_name}"

            row_id_in_file = raw_row_id % file_length + 1 if raw_row_id >= file_length else raw_row_id

        row = pd.read_csv(
            f"{self.wikiart_data_path}/wikiart/data/csv/" + file_name, skiprows=range(1, row_id_in_file), nrows=1
        )
        return row

    def __iter__(self):
        return self.generator()

    def generator(self):
        with open(os.path.join(self.file_length_map_json_path, "file_length_map.json"), "r") as file_id_map:
            map_dict = json.load(file_id_map)
            for file_name, file_length in map_dict.items():
                for chunk in pd.read_csv(
                    f"{self.wikiart_data_path}/wikiart/data/csv/" + file_name, chunksize=self.chunk_size
                ):
                    image_arrays, labels = [], []
                    image_arrays = (
                        chunk["image"]
                        # .map(lambda img: resize_img(tf.io.decode_image(ast.literal_eval(img).get("bytes")).numpy()))
                        .map(
                            lambda img: resize_img(
                                torchvision.io.decode_image(
                                    torch.tensor(np.frombuffer(ast.literal_eval(img).get("bytes"), dtype=np.uint8))
                                ),
                                self.in_shape[0],
                                self.in_shape[1],
                            )
                        ).values
                    )
                    labels = chunk[self.chosen_label].values  # .decode('utf-8')

                    yield (
                        np.asarray([change_channels(resize_img(img_arr)) for img_arr in image_arrays]),
                        np.asarray(labels),
                    )

        return self
