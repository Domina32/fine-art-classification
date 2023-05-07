from code.helpers.device_utils import Device
from code.helpers.loader_utils import get_dataloader
from code.helpers.train_utils import get_trainer

NUM_EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 0.001

device = Device().get_device()


def main():
    training_loader, testing_loader = get_dataloader(
        "wga", num_workers=12, prefetch_factor=2, pin_memory=True, batch_size=BATCH_SIZE, slice=(0, 1500)
    )

    trainer, logger = get_trainer(device, training_loader, testing_loader, overwrite_checkpoints=True)
    trainer.run(training_loader, max_epochs=NUM_EPOCHS)

    logger.close()


if __name__ == "__main__":
    main()
