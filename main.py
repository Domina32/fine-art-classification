from code.dataloader.dataloader import split_dataset
from code.dataloader.wga_generator.CustomWgaDataset import CustomWgaDataset
from code.dataloader.wikiart_generator.CustomWikiartDataset import CustomWikiartDataset
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch_directml
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torchmetrics import functional as metrics
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

dml = torch_directml.device()

BATCH_SIZE = 16


def get_dataloader(data="wga", check=False, image_save_name="./example.png", batch_size=BATCH_SIZE):
    if data == "wga":
        dataset = CustomWgaDataset(chunk_size=1)
    elif data == "wikiart":
        dataset = CustomWikiartDataset(chunk_size=1)
    else:
        raise ValueError(f"Unknown argument {data}")

    train_loader, test_loader = split_dataset(dataset, batch_size)

    if check:
        train_features, train_labels = next(iter(train_loader))
        print(f"Feature batch shape: {train_features.size()}")
        print(f"Labels batch shape: {train_labels.size()}")
        print(f"Label: {train_labels[0]}")

        if data == "wga":
            img = train_features[0]
        elif data == "wikiart":
            img = train_features[0].squeeze()
        else:
            raise ValueError(f"Unknown argument {data}")

        if image_save_name:
            plt.imshow(img, cmap="gray")
            plt.savefig(image_save_name)

    return train_loader, test_loader


loss_fn = nn.CrossEntropyLoss().to(dml)


def train_epoch(model, epoch_index, optimizer: Optimizer, training_loader: DataLoader):
    running_loss = 0.0
    last_loss = 0.0

    with tqdm(
        total=len(training_loader), position=1, bar_format="{desc}", desc=f"Batch/loss/train/accuracy: - / - / - / -"
    ) as desc:
        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in tqdm(enumerate(training_loader), unit="batch", total=len(training_loader)):
            data: Tuple[torch.Tensor, torch.Tensor] = data
            # Every data instance is an input + label pair
            inputs, labels = data
            ins, labs = inputs.to(dml), labels.to(dml)

            # Zero your gradients for every batch!
            optimizer.zero_grad()
            # Make predictions for this batch
            outputs = model(ins)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labs)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % BATCH_SIZE == BATCH_SIZE - 1:
                last_loss: float = running_loss / BATCH_SIZE  # loss per batch
                accuracy = metrics.accuracy(outputs, labs, task="multiclass", num_classes=11).to(dml)
                desc.set_description_str(f"Batch: {i + 1} Loss: {round(last_loss, 4)} Accuracy: {accuracy}")
                running_loss = 0.0

        return last_loss


NUM_EPOCHS = 5


def main():
    model_ft = resnet50(weights=ResNet50_Weights.DEFAULT).to(dml)
    optimizer = Adam(model_ft.parameters(), lr=0.001)
    in_features = model_ft.fc.in_features
    model_ft.fc = nn.Linear(in_features, 11).to(dml)
    model_ft = model_ft.train(True).to(dml)
    training_loader, test_loader = get_dataloader("wikiart", batch_size=1)

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        train_epoch(model_ft, epoch, optimizer, training_loader)

    return


if __name__ == "__main__":
    main()
