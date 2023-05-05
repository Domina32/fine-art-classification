from code.dataloader.wikiart_generator.CustomWikiartDataset import CustomWikiartDataset


def main():
    dataset = CustomWikiartDataset(chunk_size=3)
    # print(dataset.length)
    # print(dataset.data_path)

    # for i, chunk in enumerate(dataset):
    #     if i > 0:
    #         break
    #     print(chunk)

    return


if __name__ == "__main__":
    main()
