# This file does the training of the model

# Imports
import pandas as pd
from src.configs.load_config import ConfigLoader
import submit_to_kaggle
import random
import wandb
from sklearn.model_selection import GroupShuffleSplit, train_test_split

# Load config file
config = None


def main(config: ConfigLoader):
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
    df = pd.read_parquet(config.get_pp_in() + "/train_series.parquet")

    # Initialize preprocessing steps
    pp_steps, pp_s = config.get_pp_steps()
    processed = df
    # Get the preprocessing steps as a list of str to make the paths
    for i, step in enumerate(pp_steps):
        # Passes the current list because its needed to write to if the path doesnt exist
        processed = step.run(processed, pp_s[:i + 1])

    # Initialize feature engineering steps
    fe_steps, fe_s = config.get_features()
    featured_data = processed
    for i, fe_step in enumerate(fe_steps):
        # Also pass the preprocessing steps to the feature engineering step
        # to save fe for each possible pp combination
        featured_data = fe_steps[fe_step].run(processed, fe_s[:i + 1], pp_s)
        # Add feature to featured_data
        # featured_data = pd.concat([featured_data, feature], axis=1)

    # Pretrain processstep (splitting data into train and test, standardization, etc.)
    print("-------- PRETRAINING ----------")
    pretrain = config.get_pretraining()
    # Train test split
    groups = featured_data["series_id"]
    gss = GroupShuffleSplit(n_splits=1, test_size=pretrain["test_size"], random_state=42)
    train_idx, test_idx = next(gss.split(featured_data, groups=groups))
    train_data = featured_data.iloc[train_idx]
    test_data = featured_data.iloc[test_idx]

    # Print the series_id and their windows in the train_data set
    print("Train data:")
    for series_id, group in train_data.groupby("series_id"):
        print(f"Series ID: {series_id}")
        for window in group["window"].unique():
            print(f"\tWindow: {window}")

    # Print the series_id and their windows in the test_data set
    print("Test data:")
    for series_id, group in test_data.groupby("series_id"):
        print(f"Series ID: {series_id}")
        for window in group["window"].unique():
            print(f"\tWindow: {window}")
    pass
    # Initialize models
    models = config.get_models()

    # Get saved models directory from config
    store_location = config.get_model_store_loc()

    # TODO Add crossvalidation to models #107
    for model in models:
        models[model].train(featured_data)
        models[model].save(store_location + "/" + model + ".pt")

    print(models)

    # Initialize ensemble
    #    ensemble = config.get_ensemble(models)

    # TODO ADD preprocessing of data suitable for predictions #103

    # ensemble.pred(featured_data)

    # Initialize loss
    # TODO assert that every model has a loss function #67

    # TODO Hyperparameter optimization for ensembles #101
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

    # TODO Add Weights and biases to model training and record loss and acc #106

    # TODO ADD scoring to WANDB #108

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


if __name__ == "__main__":
    # Load config file
    config = ConfigLoader("config.json")

    # Train model
    main(config)

    # Create submission
    submit_to_kaggle.submit(config.get_pp_in() + "/test_series.parquet", False)
