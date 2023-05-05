import os
from pathlib import Path

from torch.utils.data import Dataset

current_dir = Path(__file__).parent
local_DATA_PATH = os.path.join(current_dir.parent.parent, 'data')

DEFAULT_IN_SHAPE = (3, 300, 300)


class CustomDataset(Dataset):
    def __init__(
        self,
        DATA_PATH=local_DATA_PATH,
        chosen_label="",
        chunk_size=1,
    ):
        self.data_path = DATA_PATH
        self.chosen_label = chosen_label
        self.chunk_size = chunk_size
        self.in_shape = DEFAULT_IN_SHAPE
        self.length = 0

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.generator()

    def generator(self):
        raise NotImplementedError()

