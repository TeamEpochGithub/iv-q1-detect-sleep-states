# This file does the training of the model

# Imports
import pandas as pd

import submit_to_kaggle
import wandb
from src.configs.load_config import ConfigLoader
from src.logger.logger import logger
from src.util.printing_utils import print_section_separator


def main(config: ConfigLoader):
    """
    Main function for training the model
    :param config: loaded config
    :return: nothing
    """
    print_section_separator("Q1 - Detect Sleep States", spacing=0, separator_length=200)
    logger.info("Start of main.py")

    # Initialize wandb
    if config.get_log_to_wandb():
        # Initialize wandb
        wandb.init(
            project='detect-sleep-states',

            config=config.get_config()
        )
        logger.info("Logging to wandb")
    else:
        logger.info("Not logging to wandb")

    # Do training here
    to_load = config.get_pp_in() + "/test_series.parquet"
    df = pd.read_parquet(to_load)

    logger.info("Loaded data from " + to_load)

    # ------------------- #
    #    Preprocessing    #
    # ------------------- #

    print_section_separator("Preprocessing", spacing=0)

    # Initialize preprocessing steps
    pp_steps, pp_s = config.get_pp_steps()
    logger.info("Preprocessing steps: " + str(pp_s))

    processed = df
    # Get the preprocessing steps as a list of str to make the paths
    for i, step in enumerate(pp_steps):
        logger.info("Running preprocessing step " + str(i) + ": " + str(pp_s[i]))
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

# Run main
main(config)

# Create submission (predict on test data)
submit_to_kaggle.submit(config.get_pp_in() + "/test_series.parquet", False)
