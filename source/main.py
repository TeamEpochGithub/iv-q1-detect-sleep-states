# This file does the training of the model

# Imports
import wandb
import random
import submit_to_kaggle
from configs.load_config import ConfigLoader
import pandas as pd
import sys

# Load config file
config = None


def train(config, wandb_on=True):

    if wandb_on:
        # Initialize wandb
        wandb.init(
            project='detect-sleep-states',

            config=config.get_config()
        )
    # Do training here

    df = pd.read_parquet(config.get_pp_in() + "/test_series.parquet")

    # Initialize preprocessing steps
    pp_steps = config.get_pp_steps()
    processed = df
    for pp_step in pp_steps:
        processed = pp_step.run(processed)

    # Initialize feature engineering steps
    fe_steps = config.get_features()
    featured_data = processed
    for fe_step in fe_steps:
        feature = fe_steps[fe_step].run(processed)
        # Add feature to featured_data
        featured_data = pd.concat([featured_data, feature], axis=1)

    # Initialize models and train them
    models = config.get_models()
    for model in models:
        models[model].train(featured_data)

    # Initialize ensemble
    ensemble = config.get_ensemble()
    ensemble.pred(featured_data)

    # Initialize loss
    loss = config.get_loss()
    loss.forward(featured_data, featured_data)

    # Initialize HPO
    hpo = config.get_hpo()
    hpo.optimize()

    # Initialize CV
    cv = config.get_cv()
    cv.run()

    # Get scoring
    scoring = config.get_scoring()
    if scoring:
        # Do scoring
        pass

    epochs = 10
    offset = random.random() / 5
    for epoch in range(2, epochs):
        acc = 1 - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset

        # log metrics to wandb
        if wandb_on:
            wandb.log({"acc": acc, "loss": loss})

    # [optional] finish the wandb run, necessary in notebooks
    if wandb_on:
        wandb.finish()


if __name__ == "__main__":
    # Load config file
    config = ConfigLoader("source/config.json")
    
    # Train model
    train(config, False)
    
    # Create submission
    submit_to_kaggle.submit(config.get_pp_in() + "/test_series.parquet", False)
