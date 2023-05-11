import warnings

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

import src.constants
from src.helpers.device_utils import Device
from src.helpers.loader_utils import get_dataloader
from src.helpers.train_utils import get_trainer

warnings.filterwarnings("ignore")

device = Device().get_device()

NUM_EPOCHS = 2
LEARNING_RATE = 0.001
BATCH_SIZE = 16

NUM_WORKERS = 0  # 12
PREFETCH_FACTOR = 2
PIN_MEMORY = True


def main():
    DATASET = "wga"
    NETWORK = "idensenet"

    training_loader, testing_loader = get_dataloader(
        DATASET,
        test_split=src.constants.TEST_SPLIT,
        batch_size=BATCH_SIZE,
        shuffle=src.constants.SHUFFLE,
        random_seed=src.constants.RANDOM_SEED,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        pin_memory=PIN_MEMORY,
    )

    training_loader = DataLoader(
        CIFAR10(root="./data", transform=ToTensor()), batch_size=BATCH_SIZE
    )

    trainer, logger = get_trainer(
        NETWORK,
        device,
        training_loader,
        testing_loader,
        learning_rate=LEARNING_RATE,
        overwrite_checkpoints=True,
        num_classes=10,
    )

    trainer.run(training_loader, max_epochs=NUM_EPOCHS)

    logger.close()


if __name__ == "__main__":
    main()
