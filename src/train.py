# This file does the training of the model

# Imports
import wandb
import random

# Load config file
config = None


def train(config):
    wandb.init(
        project='detect-sleep-states',

        config={
            'name': 'setup_run',
            'learning_rate': 0.01,
            'epochs': 10,
        }
    )
    # simulate training
    epochs = 10
    offset = random.random() / 5
    for epoch in range(2, epochs):
        acc = 1 - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset

        # log metrics to wandb
        wandb.log({"acc": acc, "loss": loss})

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()


train(config)
