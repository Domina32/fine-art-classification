import ast
import json
import os
from code.dataloader.dataloader import CustomDataset
from code.helpers.image_utils import resize_img
from pathlib import Path, PurePath

import numpy as np
import pandas as pd
import torch
import torchvision

local_FILE_LENGTH_MAP_JSON_PATH = Path(__file__).parent


class CustomWikiartDataset(CustomDataset):
    def __init__(
        self,
        WIKIART_DATA_PATH="wikiart",
        FILE_LENGTH_MAP_JSON_PATH=local_FILE_LENGTH_MAP_JSON_PATH,
        chosen_label="genre",
        chunk_size=1,
    ):
        super().__init__(chosen_label=chosen_label, chunk_size=chunk_size)

        temp_ = 0
        with open(os.path.join(FILE_LENGTH_MAP_JSON_PATH, "file_length_map.json"), "r") as file_id_map:
            map_dict = json.load(file_id_map)
            for file_length in map_dict.values():
                temp_ += file_length

        self.file_length_map_json_path = FILE_LENGTH_MAP_JSON_PATH  # type: ignore
        self.data_path = PurePath(self.data_path, WIKIART_DATA_PATH)
        self.length = temp_

    def __len__(self):
        return self.length

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

        row = pd.read_csv(f"{self.data_path}/csv/" + file_name, skiprows=range(1, row_id_in_file), nrows=1)

        return (
            resize_img(
                torchvision.io.decode_image(
                    torch.tensor(np.frombuffer(ast.literal_eval(row["image"].iloc[0]).get("bytes"), dtype=np.uint8))
                )
            ),
            row[self.chosen_label].iloc[0],
        )

    def __iter__(self):
        return self.generator()

    def generator(self):
        with open(os.path.join(self.file_length_map_json_path, "file_length_map.json"), "r") as file_id_map:
            map_dict = json.load(file_id_map)
            for file_name, file_length in map_dict.items():
                for chunk in pd.read_csv(f"{self.data_path}/csv/" + file_name, chunksize=self.chunk_size):
                    image_arrays, labels = [], []
                    image_arrays = (
                        chunk["image"]
                        # .map(lambda img: resize_img(tf.io.decode_image(ast.literal_eval(img).get("bytes")).numpy()))
                        .map(
                            lambda img: resize_img(
                                torchvision.io.decode_image(
                                    torch.tensor(np.frombuffer(ast.literal_eval(img).get("bytes"), dtype=np.uint8))
                                )
                            )
                        ).values
                    )
                    labels = chunk[self.chosen_label].values  # .decode('utf-8')

                    yield (
                        image_arrays,
                        np.asarray(labels),
                    )

        return self
