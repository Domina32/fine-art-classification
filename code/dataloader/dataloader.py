from torch.utils.data import Dataset

local_DATA_PATH = "/home/rudovc/Git/dome/fine_art_classification/data"  # "./data"


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
        self.in_shape = (300, 300, 3)
        self.length = 0

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.generator()

    def generator(self):
        raise NotImplementedError()
