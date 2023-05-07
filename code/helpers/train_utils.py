from code.models.model import Net
from typing import Literal, Optional, Union

from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
from ignite.engine import Engine, Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.metrics import Accuracy, Loss
from torch import device
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

NUM_EPOCHS = 5
LEARNING_RATE = 0.001


# Score function to return current value of any metric we defined above in val_metrics
def score_function(engine):
    return engine.state.metrics["accuracy"]


def log_progress(engine: Engine, engine_type: Optional[Union[Literal["Training"], Literal["Validation"]]]):
    if engine_type == "Training":
        output = engine.state.output
        print(
            f"{engine_type} - Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {output:.2f}",
            end="\r",
        )
    else:
        output = f"Avg accuracy: {engine.state.metrics['accuracy']:.2f}" if "accuracy" in engine.state.metrics else ""
        print(
            f"{engine_type} - Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] {output}",
            end="\r",
        )


def log_results(
    trainer: Engine,
    evaluator: Engine,
    loader: DataLoader,
    evaluator_type: Union[Literal["Training"], Literal["Validation"]],
):
    evaluator.run(loader)
    metrics = evaluator.state.metrics
    print(
        f"{evaluator_type} Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}",
    )


def get_logger(train_evaluator: Engine, test_evaluator: Engine, trainer: Engine):
    # Define a Tensorboard logger
    tb_logger = TensorboardLogger(log_dir="tb-logger")

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=100),
        tag="training",
        output_transform=lambda loss: {"batch_loss": loss},
    )

    # Attach handler for plotting both evaluators' metrics after every epoch completes
    for tag, evaluator in [("training", train_evaluator), ("validation", test_evaluator)]:
        tb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names="all",
            global_step_transform=global_step_from_engine(trainer),
        )

    return tb_logger


def get_trainer(
    device: device,
    train_loader: DataLoader,
    test_loader: DataLoader,
    learning_rate=LEARNING_RATE,
    overwrite_checkpoints=False,
) -> tuple[Engine, TensorboardLogger]:
    model = Net().freeze().to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_function = CrossEntropyLoss()
    trainer = create_supervised_trainer(model, optimizer, loss_function, device)
    metrics = {
        "accuracy": Accuracy(),
        "loss": Loss(loss_function),
    }
    training_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    test_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

    trainer.add_event_handler(Events.ITERATION_COMPLETED, log_progress, "Training")
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        log_results,
        training_evaluator,
        train_loader,
        evaluator_type="Training",
    )

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, log_results, test_evaluator, test_loader, evaluator_type="Validation"
    )

    for evaluator, name in (training_evaluator, "Train evaluator"), (test_evaluator, "Val evaluator"):
        evaluator.add_event_handler(Events.ITERATION_COMPLETED, log_progress, name)
        pass

    # Checkpoint to store n_saved best models wrt score function
    model_checkpoint = ModelCheckpoint(
        "checkpoint",
        n_saved=2,
        filename_prefix="best",
        score_function=score_function,
        score_name="accuracy",
        global_step_transform=global_step_from_engine(trainer),  # helps fetch the trainer's state
        require_empty=not overwrite_checkpoints,
    )

    # Save the model after every epoch of test_evaluator is completed
    test_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

    logger = get_logger(train_evaluator=training_evaluator, test_evaluator=test_evaluator, trainer=trainer)

    return trainer, logger
