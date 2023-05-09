from constants import (
    TEST_SPLIT,
    SHUFFLE,
    RANDOM_SEED,
    # BATCH_SIZE,
    # NUM_EPOCHS,
    # LEARNING_RATE,
    # PREFETCH_FACTOR,
    # NUM_WORKERS,
    # PIN_MEMORY,
)

from code.helpers.device_utils import Device
from code.helpers.loader_utils import get_dataloader
from code.helpers.train_utils import get_trainer


device = Device().get_device()


NUM_EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 16

NUM_WORKERS = 12
PREFETCH_FACTOR = 2
PIN_MEMORY = True


def main():
    DATASET = "wikiart"
    NETWORK = "resnet"

    training_loader, testing_loader = get_dataloader(
        DATASET,
        test_split=TEST_SPLIT,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        random_seed=RANDOM_SEED,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        pin_memory=PIN_MEMORY,
    )

    trainer, logger = get_trainer(
        NETWORK,
        device,
        training_loader,
        testing_loader,
        learning_rate=LEARNING_RATE,
        overwrite_checkpoints=True,
    )

    trainer.run(training_loader, max_epochs=NUM_EPOCHS)

    logger.close()


if __name__ == "__main__":
    main()
