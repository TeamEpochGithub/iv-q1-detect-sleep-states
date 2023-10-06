# This file does the training of the model
import json

import pandas as pd
import wandb

from src import submit_to_kaggle
from src.configs.load_config import ConfigLoader
from src.get_processed_data import get_processed_data
from src.logger.logger import logger
from src.pre_train.train_test_split import train_test_split, split_on_labels
from src.score.doscoring import compute_scores
from src.util.printing_utils import print_section_separator
from src.util.submissionformat import to_submission_format


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

    # ------------------------------------------- #
    #    Preprocessing and feature Engineering    #
    # ------------------------------------------- #

    print_section_separator("Preprocessing and feature engineering", spacing=0)

    featured_data = get_processed_data(config, training=True, save_output=True)

    # ------------------------- #
    #         Pre-train         #
    # ------------------------- #

    print_section_separator("Pre-train", spacing=0)

    logger.info("Get pretraining parameters from config...")
    pretrain = config.get_pretraining()

    logger.info("Obtained pretrain parameters from config " + str(pretrain))
    # Split data into train and test
    # Use numpy.reshape to turn the data into a 3D tensor with shape (window, n_timesteps, n_features)

    logger.info("Splitting data into train and test...")
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(featured_data,
                                                                             test_size=pretrain["test_size"],
                                                                             standardize_method=pretrain["standardize"])

    # Give data shape in terms of (features (in_channels), window_size))
    data_shape = (X_train.shape[2], X_train.shape[1])

    logger.info("X Train data shape (size, window_size, features): " + str(
        X_train.shape) + " and y train data shape (size, window_size, features): " + str(y_train.shape))
    logger.info("X Test data shape (size, window_size, features): " + str(
        X_test.shape) + " and y test data shape (size, window_size, features): " + str(y_test.shape))

    # TODO Cross validation should be part of each model
    cv = 0
    if "cv" in pretrain:
        cv = config.get_cv()

    # ------------------------- #
    #          Training         #
    # ------------------------- #

    print_section_separator("Training", spacing=0)

    # Initialize models
    store_location = config.get_model_store_loc()
    logger.info("Model store location: " + store_location)

    # Initialize models
    logger.info("Initializing models...")
    models = config.get_models(data_shape)

    # TODO Add crossvalidation to models and hyperparameter optimization #107, #101
    for i, model in enumerate(models):
        logger.info("Training model " + str(i) + ": " + model)
        models[model].train(X_train, X_test, y_train, y_test)
        models[model].save(store_location + "/" + model + ".pt")

    # Store optimal models
    for i, model in enumerate(models):
        models[model].save(store_location + "/optimal_" + model + ".pt")

    # ------------------------------------------------------- #
    #                    Train for submission                 #
    # ------------------------------------------------------- #

    print_section_separator("Train for submission", spacing=0)

    if config.get_train_for_submission():
        logger.info("Retraining models for submission")

        # Retrain all models with optimal parameters
        for i, model in enumerate(models):
            models[model].load(store_location + "/optimal_" + model + ".pt", only_hyperparameters=True)
            logger.info("Retraining model " + str(i) + ": " + model)
            models[model].train_full(*split_on_labels(featured_data))
            models[model].save(store_location + "/submit_" + model + ".pt")
    else:
        logger.info("Not training best model for submission")

    # ------------------------- #
    #          Ensemble         #
    # ------------------------- #

    print_section_separator("Ensemble", spacing=0)
    # TODO Add crossvalidation to models #107
    ensemble = config.get_ensemble(models)

    # Initialize loss
    # TODO assert that every model has a loss function #67

    # ------------------------------------------------------- #
    #          Hyperparameter optimization for ensemble       #
    # ------------------------------------------------------- #

    print_section_separator("Hyperparameter optimization for ensemble", spacing=0)
    # TODO Hyperparameter optimization for ensembles
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
    #                    Scoring                              #
    # ------------------------------------------------------- #

    print_section_separator("Scoring", spacing=0)

    scoring = config.get_scoring()
    if scoring:
        logger.info("Making predictions with ensemble on test data")
        predictions = ensemble.pred(X_test)

        logger.info("Formatting predictions...")

        # for each window get the series id and step offset
        window_info = (featured_data.iloc[test_idx][['series_id', 'window', 'step']]
                       .groupby(['series_id', 'window'])
                       .apply(lambda x: x.iloc[0]))
        submission = to_submission_format(predictions, window_info)

        # get only the test series data from the solution
        test_series_ids = window_info['series_id'].unique()
        with open('./series_id_encoding.json', 'r') as f:
            encoding = json.load(f)
        decoding = {v: k for k, v in encoding.items()}
        test_series_ids = [decoding[sid] for sid in test_series_ids]

        solution = (pd.read_csv(config.get_train_events_path())
                    .groupby('series_id')
                    .filter(lambda x: x['series_id'].iloc[0] in test_series_ids))

        logger.info("Start scoring test predictions...")
        compute_scores(submission, solution)  # TODO Add scoring to WANDB #103
    else:
        logger.info("Not scoring")

    # TODO Add Weights and biases to model training and record loss and acc #106

    # TODO ADD scoring to WANDB #108

    # [optional] finish the wandb run, necessary in notebooks
    if config.get_log_to_wandb():
        wandb.finish()
        logger.info("Finished logging to wandb")


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install()

    # Load config file
    config = ConfigLoader("config.json")

    # Run main
    main(config)

    # Create submission
    submit_to_kaggle.submit(config, False)
