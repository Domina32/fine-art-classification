from code.dataloader.wga_generator.CustomWgaDataset import CustomWgaDataset
from code.dataloader.wikiart_generator.CustomWikiartDataset import \
    CustomWikiartDataset


def main():
    wga_dataset = CustomWgaDataset(chunk_size=1)
    wikiart_dataset = CustomWikiartDataset(chunk_size=1)
    

    for i, chunk in enumerate(wikiart_dataset):
        if i > 2:
            break

        if i == 0:
            print(chunk)
            pass

    return


if __name__ == "__main__":
    main()
