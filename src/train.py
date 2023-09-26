# This file does the training of the model

# Imports
import wandb
import random
from src.configs.load_config import ConfigLoader
import pandas as pd

# Load config file
config = None


def train(config):

    # Initialize the path used for checking
    # If pp already exists
    print(config)
    # Initialize wandb
    if config.get_log_to_wandb():
        # Initialize wandb
        wandb.init(
            project='detect-sleep-states',

            config=config.get_config()
        )
    else:
        print("Not logging to wandb.")
    # Do training here

    df = pd.read_parquet(config.get_pp_in() + "/test_series.parquet")

    # Initialize preprocessing steps
    pp_steps, pp_s = config.get_pp_steps()
    processed = df
    # Get the preprocessing steps as a list of str to make the paths
    for i, step in enumerate(pp_steps):
        # Passes the current list because its needed to write to if the path doesnt exist
        processed = step.run(processed, pp_s[:i+1])

    # Initialize feature engineering steps
    fe_steps, fe_s = config.get_features()
    featured_data = processed
    for i, fe_step in enumerate(fe_steps):
        # Also pass the preprocessing steps to the feature engineering step
        # to save fe for each possible pp combination
        feature = fe_steps[fe_step].run(processed, fe_s[:i+1], pp_s)
        # Add feature to featured_data
        featured_data = pd.concat([featured_data, feature], axis=1)

    # Initialize models
    models = config.get_models()

    for model in models:
        models[model].train(featured_data)

    print(models)

    # Initialize ensemble
    ensemble = config.get_ensemble(models)
    ensemble.pred(featured_data)

    # Initialize loss
    #TODO assert that every model has a loss function
    #TODO assert that the loss function is forwarded to the model

    ensemble_loss = config.get_loss()
    loss.forward(featured_data, featured_data)


    #TODO Hyperparameter optimization for ensembles
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
        if config.get_log_to_wandb():
            wandb.log({"acc": acc, "loss": loss})

    # [optional] finish the wandb run, necessary in notebooks
    if config.get_log_to_wandb():
        wandb.finish()


config = ConfigLoader("src/configs/config.json")
train(config)
