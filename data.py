import argparse
import os
import zipfile

import pandas as pd
import requests
from datasets import Dataset, DatasetDict, load_dataset
from pandas import DataFrame

from src.dataloader.wga_generator.custom_wga_dataset import make_wga_dataset_chunks, merge_wga_chunks
from src.helpers.image_utils import transform_URL
from src.helpers.loader_utils import paintings_label_mapping

WGA_URL = "https://www.wga.hu/database/download/data_xls.zip"
WGA_PATH = os.path.join(".", "data", "wga")
WIKIART_PATH = os.path.join(".", "data", "wikiart")


def export_wikiart_labels():
    dataset = load_dataset("huggan/wikiart", cache_dir="./data/wikiart/labels")
    if isinstance(dataset, DatasetDict):
        labels = dataset["train"].remove_columns("image")
        labels.save_to_disk("./data/wikiart/labels")
    else:
        raise TypeError(f"Expected dataset to be {DatasetDict}, got {type(dataset)} instead")


def save_wikiart_labels_as_dataframe():
    labels = Dataset.load_from_disk("./data/wikiart/labels")
    df = labels.to_pandas()
    if isinstance(df, DataFrame):
        df.to_csv("./data/wikiart/labels/labels.csv", index=False)
    else:
        raise TypeError(f"Expected df to be {DataFrame}, got {type(df)} instead")


def get_wga():
    zip_path = os.path.join(WGA_PATH, "wga.zip")
    catalog_excel_path = os.path.join(WGA_PATH, "catalog.xlsx")
    csv_path = os.path.join(WGA_PATH, "paintings.csv")
    temp_csv_path = os.path.join(WGA_PATH, "temp.csv")
    images_data_path = os.path.join(WGA_PATH, "wga_images.npy")
    labels_data_path = os.path.join(WGA_PATH, "wga_labels.npy")

    if not os.path.exists(images_data_path) or not os.path.exists(labels_data_path):
        if not os.path.exists(zip_path) and not os.path.exists(catalog_excel_path) and not os.path.exists(csv_path):
            response = requests.get(WGA_URL, allow_redirects=True)
            with open(zip_path, "wb") as file:
                file.write(response.content)

        if not os.path.exists(catalog_excel_path) and not os.path.exists(csv_path):
            with zipfile.ZipFile(zip_path, "r") as zip:
                zip.extractall(WGA_PATH)

        if not os.path.exists(csv_path):
            data = pd.read_excel(catalog_excel_path)
            data["new_URL"] = data["URL"].apply(transform_URL)
            data.drop(
                ["BORN-DIED", "TITLE", "DATE", "TECHNIQUE", "LOCATION", "SCHOOL", "TIMEFRAME"], axis=1, inplace=True
            )
            paintings = data.loc[data["FORM"] == "painting"]
            paintings.reset_index(inplace=True, drop=True)
            paintings.to_csv(csv_path, index=False)

        if not os.path.exists(temp_csv_path):
            paintings_label_mapping()

        if len(os.listdir(os.path.join(WGA_PATH, "data_chunks"))) < 30:
            make_wga_dataset_chunks()

        merge_wga_chunks()


def get_wikiart():
    load_dataset("huggan/wikiart", cache_dir="./data/wikiart/dataset")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--get", choices=["wga", "wikiart"], help="Download selected datasets", nargs="*", required=False
    )
    parser.add_argument(
        "--clean", choices=["wga", "wikiart"], help="Erase selected datasets", nargs="*", required=False
    )
    parser.add_argument(
        "--export", choices=["wikiart"], help="Export selected dataset's labels", nargs="*", required=False
    )
    args = parser.parse_args()

    try:
        os.makedirs(os.path.join(WGA_PATH, "labels"))
        print(f"Created path {WGA_PATH}")
    except OSError:
        pass
    try:
        os.makedirs(os.path.join(WGA_PATH, "data_chunks"))
        print(f"Created path {WGA_PATH}")
    except OSError:
        pass
    try:
        wikiart_dataset_path = os.path.join(WIKIART_PATH, "dataset")
        os.makedirs(wikiart_dataset_path)
        print(f"Created path {wikiart_dataset_path}")
    except OSError:
        pass
    try:
        wikiart_labels_path = os.path.join(WIKIART_PATH, "labels")
        os.makedirs(wikiart_labels_path)
        print(f"Created path {wikiart_labels_path}")
    except OSError:
        pass

    if args.get:
        if "wga" in args.get:
            get_wga()

        if "wikiart" in args.get:
            get_wikiart()

    if args.export and "wikiart" in args.export:
        export_wikiart_labels()


if __name__ == "__main__":
    main()
