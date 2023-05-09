from constants import BATCH_SIZE, NUM_EPOCHS, TEST_SPLIT, SHUFFLE, RANDOM_SEED

from code.helpers.device_utils import Device
from code.helpers.loader_utils import get_dataloader
from code.helpers.train_utils import get_trainer


device = Device().get_device()


def main():
    training_loader, testing_loader = get_dataloader(
        "wikiart",
        test_split=TEST_SPLIT,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        random_seed=RANDOM_SEED,
        num_workers=12,
        prefetch_factor=2,
        pin_memory=True,
    )

    trainer, logger = get_trainer(
        device, training_loader, testing_loader, overwrite_checkpoints=True
    )
    trainer.run(training_loader, max_epochs=NUM_EPOCHS)

    logger.close()


def train_wga_resnet():
    pass


def train_wikiart_resnet():
    pass


def train_wga_densenet():
    pass


def train_wikiart_densenet():
    pass


if __name__ == "__main__":
    main()
