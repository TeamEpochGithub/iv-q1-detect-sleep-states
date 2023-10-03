# This file does the training of the model

# Imports
import pandas as pd
import wandb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import gc
import numpy as np

from src import submit_to_kaggle
from src.configs.load_config import ConfigLoader
from src.logger.logger import logger
from src.util.printing_utils import print_section_separator


def main(config: ConfigLoader) -> None:
    """
    Main function for training the model

    :param config: loaded config
    """
    print_section_separator("Q1 - Detect Sleep States - Kaggle", spacing=0)
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
    to_load = config.get_pp_in() + "/train_series.parquet"
    df = pd.read_parquet(to_load)

    logger.info("Loaded data from " + to_load)

    # ------------------- #
    #    Preprocessing    #
    # ------------------- #

    print_section_separator("Preprocessing", spacing=0)

    # Initialize preprocessing steps
    print("-------- PREPROCESSING ----------")
    pp_steps, pp_s = config.get_pp_steps()
    logger.info("Preprocessing steps: " + str(pp_s))

    processed = df
    # Get the preprocessing steps as a list of str to make the paths
    for i, step in enumerate(pp_steps):
        logger.info("Running preprocessing step " + str(i) + ": " + str(pp_s[i]))
        # Passes the current list because it's needed to write to if the path doesn't exist
        processed = step.run(processed, pp_s[:i + 1])

    # ------------------------- #
    #    Feature Engineering    #
    # ------------------------- #

    print_section_separator("Feature engineering", spacing=0)

    fe_steps, fe_s = config.get_features()
    logger.info("Feature engineering steps: " + str(fe_s))
    featured_data = processed
    for i, fe_step in enumerate(fe_steps):
        logger.info("Running feature engineering step " + str(i) + ": " + str(fe_s[i]))
        # Passes the current list because it's needed to write to if the path doesn't exist
        featured_data = fe_steps[fe_step].run(processed, fe_s[:i + 1], pp_s)

    # Pretrain processstep (splitting data into train and test, standardization, etc.)
    print("-------- PRETRAINING ----------")
    pretrain = config.get_pretraining()


    # Use numpy.reshape to turn the data into a 3D tensor with shape (window, n_timesteps, n_features)
    exclude_x = ['timestamp', 'window', 'step', 'awake']
    keep_y_train_columns = []
    if 'awake' in featured_data.columns:
        keep_y_train_columns.append('awake')
    x_columns = featured_data.columns.drop(exclude_x)
    X_featured_data = featured_data[x_columns].to_numpy().reshape(-1, 17280, len(x_columns))
    Y_featured_data = featured_data[keep_y_train_columns].to_numpy().reshape(-1, 17280, len(keep_y_train_columns))

    # Standardize data
    exclude_columns = ['series_id', 'timestamp',
                       'window', 'step', 'awake']  # 'onset', 'wakeup']
    scaler = None
    if pretrain["standardize"] == "standard":
        scaler = StandardScaler()
    elif pretrain["standardize"] == "minmax":
        scaler = MinMaxScaler()

    # Drop columns to exclude from standardization
    data_to_standardize = featured_data.drop(columns=exclude_columns)
    data_to_standardize = scaler.fit_transform(data_to_standardize)
    # Add columns back to standardized data
    standardized_data = pd.DataFrame(
        data_to_standardize, columns=featured_data.columns.drop(exclude_columns))
    featured_data = pd.concat(
        [featured_data[exclude_columns], standardized_data], axis=1)
    print("Data standardized")

    # Train test split on series id
    # Check if test size key exists in pretrain
    train_data: pd.DataFrame = None
    test_data = None
    if "test_size" in pretrain:
        groups = featured_data["series_id"]
        gss = GroupShuffleSplit(
            n_splits=1, test_size=pretrain["test_size"], random_state=42)
        train_idx, test_idx = next(gss.split(featured_data, groups=groups))
        train_data = featured_data.iloc[train_idx]
        test_data = featured_data.iloc[test_idx]
        print("Data split into train and test")
    else:
        cv = config.get_cv()
        print("Crossvalidation will be used instead of train test split")

    # ------------------------- #
    #         Pre-train         #
    # ------------------------- #

    print_section_separator("Pre-train", spacing=0)

    # ------------------------- #
    #          Training         #
    # ------------------------- #

    print_section_separator("Training", spacing=0)

    # Initialize models
    print("-------- TRAINING MODELS ----------")
    models = config.get_models()
    for model in models:
        models[model].train(train_data)

    store_location = config.get_model_store_loc()
    logger.info("Model store location: " + store_location)

    # TODO Add crossvalidation to models #107
    for i, model in enumerate(models):
        logger.info("Training model " + str(i) + ": " + model)
        models[model].train(featured_data)
        models[model].save(store_location + "/" + model + ".pt")

    # Get saved models directory from config
    store_location = config.get_model_store_loc()

    # ------------------------- #
    #          Ensemble         #
    # ------------------------- #

    print_section_separator("Ensemble", spacing=0)
    # TODO Add crossvalidation to models #107
    print("-------- ENSEMBLING ----------")
    ensemble = config.get_ensemble(models)

    # TODO ADD preprocessing of data suitable for predictions #103

    ensemble.pred(test_data)

    # Initialize loss
    # TODO assert that every model has a loss function #67

    # ------------------------------------------------------- #
    #          Hyperparameter optimization for ensemble       #
    # ------------------------------------------------------- #

    print_section_separator("Hyperparameter optimization for ensemble", spacing=0)
    # TODO Hyperparameter optimization for ensembles #101
    hpo = config.get_hpo()
    hpo.optimize()

    # ------------------------------------------------------- #
    #          Cross validation optimization for ensemble     #
    # ------------------------------------------------------- #
    print_section_separator("Cross validation for ensemble", spacing=0)
    # Initialize CV
    cv = config.get_cv()
    cv.run()

    # ------------------------------------------------------- #
    #                    Train for submission                 #
    # ------------------------------------------------------- #

    print_section_separator("Train for submission", spacing=0)

    # TODO Mark best model from CV/HPO and train it on all data
    if config.get_train_for_submission():
        logger.info("Training best model for submission")
        best_model = None
        best_model.train(featured_data)
        # Add submit in name for saving
        best_model.save(store_location + "/submit_" + best_model.name + ".pt")
    else:
        logger.info("Not training best model for submission")

    # ------------------------------------------------------- #
    #                    Scoring                              #
    # ------------------------------------------------------- #

    print_section_separator("Scoring", spacing=0)

    scoring = config.get_scoring()
    if scoring:
        logger.info("Start scoring...")
        # Do scoring
        pass
    else:
        logger.info("Not scoring")

    # TODO Add Weights and biases to model training and record loss and acc #106

    # TODO ADD scoring to WANDB #108

    # [optional] finish the wandb run, necessary in notebooks
    if config.get_log_to_wandb():
        wandb.finish()
        logger.info("Finished logging to wandb")


if __name__ == "__main__":
    # Load config file
    config = ConfigLoader("config.json")

    # Run main
    main(config)

    # Create submission
    submit_to_kaggle.submit(config, config.get_pp_in() + "/test_series.parquet", False)
