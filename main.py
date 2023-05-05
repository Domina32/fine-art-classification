from code.dataloader.wga_generator.CustomWgaDataset import CustomWgaDataset
from code.dataloader.wikiart_generator.CustomWikiartDataset import CustomWikiartDataset


def main():
    wga_dataset = CustomWgaDataset(chunk_size=3)
    wikiart_dataset = CustomWikiartDataset(chunk_size=3)

    # for i, chunk in enumerate(wga_dataset):
    #     if i > 2:
    #         break

    #     if i == 0:
    #         image = np.swapaxes(np.swapaxes(chunk[0][0], 0, 1), 1, 2)
    #         pass

    return


if __name__ == "__main__":
    main()
