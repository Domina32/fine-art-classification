import os
import sys

sys.path.append(os.path.abspath(os.path.join(".", "src", "models", "IDensenet")))

import warnings

import numpy as np
import pandas as pd

from src.helpers.device_utils import Device
from src.helpers.image_utils import label_mapping, url_to_numpy

warnings.filterwarnings("ignore")

device = Device().get_device()

NUM_EPOCHS = 2
LEARNING_RATE = 0.001
BATCH_SIZE = 16

NUM_WORKERS = 2  # 12
PREFETCH_FACTOR = 2  # Must be 0 if NUM_WORKERS is also 0
PIN_MEMORY = True


def main():
    DATASET = "wikiart"
    NETWORK = "idensenet"

    # training_loader, testing_loader = get_dataloader(
    #     DATASET,
    #     test_split=src.constants.TEST_SPLIT,
    #     batch_size=BATCH_SIZE,
    #     shuffle=src.constants.SHUFFLE,
    #     random_seed=src.constants.RANDOM_SEED,
    #     num_workers=NUM_WORKERS,
    #     prefetch_factor=PREFETCH_FACTOR,
    #     pin_memory=PIN_MEMORY,
    # )

    # trainer, logger = get_trainer(
    #     NETWORK,
    #     device,
    #     training_loader,
    #     testing_loader,
    #     learning_rate=LEARNING_RATE,
    #     overwrite_checkpoints=True,
    #     num_classes=10,
    # )

    # trainer.run(training_loader, max_epochs=NUM_EPOCHS)

    # logger.close()
    merge_wga_chunks()


def make_wga_dataset_chunks():
    print("Starting...")
    path = "./data/wga/"
    paintings = pd.read_csv(os.path.join(path, "paintings.csv"))
    paintings["encoded_TYPE"] = paintings["TYPE"].map(label_mapping)
    print("Paintings label mapping done")

    temp = paintings[["new_URL", "encoded_TYPE"]].copy()
    temp.to_csv(os.path.join(path, "temp.csv"), index=False)

    NUMBER_OF_ARRAYS = 30
    temps = np.array_split(temp, NUMBER_OF_ARRAYS)
    path = "./data/wga/data_chunks"

    print("Starting data chunks")
    for i, t in enumerate(temps):
        if i <= -1:
            continue

        pom = t["new_URL"].map(lambda img: url_to_numpy(img))
        t.insert(1, "image_array", pom)

        t.dropna(inplace=True)

        A = t["image_array"].values
        L = t["encoded_TYPE"].values

        images_name = f"wga_imgs_{i}.npy"
        np.save(os.path.join(path, images_name), A)

        labels_name = f"wga_lbls_{i}.npy"
        np.save(os.path.join(path, labels_name), L)

        print(f"Saved data chunk index: {i}/{NUMBER_OF_ARRAYS}")


def merge_wga_chunks():
    NUMBER_OF_ARRAYS = 30
    path = "./data/wga/data_chunks"

    x = np.load(os.path.join(path, "wga_imgs_0.npy"), allow_pickle=True)
    all_images = np.array([np.squeeze(xx) for xx in x])

    all_labels = np.load(os.path.join(path, "wga_lbls_0.npy"), allow_pickle=True)

    for i in range(1, NUMBER_OF_ARRAYS):
        x = np.load(os.path.join(path, f"wga_imgs_{i}.npy"), allow_pickle=True)
        images_array = np.array([np.squeeze(xx) for xx in x])
        all_images = np.append(all_images, images_array, axis=0)

        labels_array = np.load(os.path.join(path, f"wga_lbls_{i}.npy"), allow_pickle=True)
        all_labels = np.append(all_labels, labels_array)

    path = "./data/wga/"

    np.save(os.path.join(path, "wga_images.npy"), all_images)
    np.save(os.path.join(path, "wga_labels.npy"), all_labels)


if __name__ == "__main__":
    main()
