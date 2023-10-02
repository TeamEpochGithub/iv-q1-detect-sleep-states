# This file does the training of the model

# Imports
import pandas as pd
from src.configs.load_config import ConfigLoader
from src import submit_to_kaggle
import wandb

# Load config file
config = None


def main(config: ConfigLoader):

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
        processed = step.run(processed, pp_s[:i + 1])

    # Initialize feature engineering steps
    fe_steps, fe_s = config.get_features()
    featured_data = processed
    for i, fe_step in enumerate(fe_steps):
        # Also pass the preprocessing steps to the feature engineering step
        # to save fe for each possible pp combination
        featured_data = fe_steps[fe_step].run(processed, fe_s[:i + 1], pp_s)

    # TODO Add pretrain processstep (splitting data into train and test, standardization, etc.) #103

    # Initialize models
    models = config.get_models()

    # Get saved models directory from config
    store_location = config.get_model_store_loc()

    # TODO Add crossvalidation to models #107
    ensemble = config.get_ensemble(models)

    # TODO ADD preprocessing of data suitable for predictions #103

    ensemble.pred(featured_data)

    # Initialize loss
    # TODO assert that every model has a loss function #67

    # TODO Hyperparameter optimization for ensembles #101
    hpo = config.get_hpo()
    hpo.optimize()

    # Initialize CV
    cv = config.get_cv()
    cv.run()

    # Train model fully on all data
    # TODO Mark best model from CV/HPO and train it on all data
    if config.get_train_for_submission():
        best_model = None
        best_model.train(featured_data)
        # Add submit in name for saving
        best_model.save(store_location + "/submit_" + best_model.name + ".pt")

    # Get scoring
    scoring = config.get_scoring()
    if scoring:
        # Do scoring
        pass

    # TODO Add Weights and biases to model training and record loss and acc #106

    # TODO ADD scoring to WANDB #108

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
