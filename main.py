# This file does the training of the model
import json
import os

import numpy as np
import pandas as pd

import wandb
from src.configs.load_config import ConfigLoader
from src.get_processed_data import get_processed_data
from src.logger.logger import logger
from src.pretrain.pretrain import Pretrain
from src.score.compute_score import log_scores_to_wandb, compute_score_full, compute_score_clean
from src.score.nan_confusion import compute_nan_confusion_matrix
from src.score.visualize_preds import plot_preds_on_series
from src.util.hash_config import hash_config
from src.util.printing_utils import print_section_separator
from src.util.submissionformat import to_submission_format


def main(config: ConfigLoader) -> None:
    """
    Main function for training the model

    :param config: loaded config
    """
    print_section_separator("Q1 - Detect Sleep States - Kaggle", spacing=0)
    logger.info("Start of main.py")

    config_hash = hash_config(config.get_config(), length=16)
    logger.info("Config hash encoding: " + config_hash)

    # Initialize wandb
    if config.get_log_to_wandb():
        # Initialize wandb
        wandb.init(
            project='detect-sleep-states',
            name=config_hash,
            config=config.get_config()
        )
        wandb.run.summary.update(config.get_config())
        logger.info(f"Logging to wandb with run id: {config_hash}")
    else:
        logger.info("Not logging to wandb")

    # Predict with CPU
    pred_cpu = config.get_pred_with_cpu()
    if pred_cpu:
        logger.info("Predicting with CPU")
    else:
        logger.info("Predicting with GPU")
    # ------------------------------------------- #
    #    Preprocessing and feature Engineering    #
    # ------------------------------------------- #

    print_section_separator("Preprocessing and feature engineering", spacing=0)

    featured_data = get_processed_data(config, training=True, save_output=True)

    # ------------------------ #
    #         Pretrain         #
    # ------------------------ #

    print_section_separator("Pretrain", spacing=0)

    logger.info("Get pretraining parameters from config and initialize pretrain")
    pretrain: Pretrain = config.get_pretraining()

    logger.info("Pretraining with scaler " + str(pretrain.scaler.kind) + " and test size of " + str(pretrain.test_size))

    # Split data into train/test and validation
    # Use numpy.reshape to turn the data into a 3D tensor with shape (window, n_timesteps, n_features)
    logger.info("Splitting data into train and test sets")

    X_train, X_test, y_train, y_test, train_idx, test_idx, groups = pretrain.pretrain_split(
        featured_data)

    # Give data shape in terms of (features (in_channels), window_size))
    data_shape = (X_train.shape[2], X_train.shape[1])

    logger.info("X train data shape (size, window_size, features): " + str(
        X_train.shape) + " and y train data shape (size, window_size, features): " + str(y_train.shape))
    logger.info("X test data shape (size, window_size, features): " + str(
        X_test.shape) + " and y test data shape (size, window_size, features): " + str(y_test.shape))

    # ------------------------- #
    # Cross Validation Training #
    # ------------------------- #

    print_section_separator("Cross Validation Training", spacing=0)

    # Initialize models
    store_location = config.get_model_store_loc()
    logger.info("Model store location: " + store_location)

    # Initialize models
    logger.info("Initializing models...")
    models = config.get_models(data_shape)

    # Hash of concatenated string of preprocessing, feature engineering and pretraining
    initial_hash = hash_config(config.get_pp_fe_pretrain(), length=5)

    for i, model in enumerate(models):
        # Get filename of model
        model_filename = store_location + "/" + model + "-" + initial_hash + models[model].hash + ".pt"
        # If this file exists, load instead of start training
        if os.path.isfile(model_filename):
            logger.info("Found existing trained model " + str(i) + ": " + model + " with location " + model_filename)
            models[model].load(model_filename, only_hyperparameters=False)
        else:
            logger.info("Training model " + str(i) + ": " + model)
            cv = config.get_cv()
            # TODO Implement hyperparameter optimization #101
            # It now only saves the trained model from the last fold
            scores: np.ndarray = cv.cross_validate(models[model], X_train, y_train, groups=groups,
                                                   scoring_params={"featured_data": featured_data,
                                                                   "train_validate_idx": train_idx,
                                                                   "downsampling_factor": pretrain.downsampler.factor,
                                                                   "window_size": pretrain.window_size})
            models[model].save(model_filename)
            logger.info(
                f"Done training model {i}: {model} with CV scores of {scores} and mean score of {np.round(np.mean(scores))}")

    # Store optimal models
    for i, model in enumerate(models):
        model_filename_opt = store_location + "/optimal_" + model + "-" + initial_hash + models[model].hash + ".pt"
        models[model].save(model_filename_opt)

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

    print_section_separator(
        "Hyperparameter optimization for ensemble", spacing=0)
    # TODO Hyperparameter optimization for ensembles
    hpo = config.get_hpo()
    hpo.optimize()

    # ------------------------------------------------------- #
    #          Cross validation optimization for ensemble     #
    # ------------------------------------------------------- #
    # print_section_separator("Cross validation for ensemble", spacing=0)

    # ------------------------------------------------------- #
    #                    Scoring                              #
    # ------------------------------------------------------- #

    print_section_separator("Scoring", spacing=0)

    scoring = config.get_scoring()
    if scoring:
        logger.info("Making predictions with ensemble on test data")
        predictions = ensemble.pred(X_test)

        logger.info("Formatting predictions...")

        # TODO simplify this
        # for each window get the series id and step offset
        # FIXME window_info for some series starts with a very large step instead of 0,
        #  close to the uint32 limit of 4294967295, likely due to integer underflow
        window_info = (featured_data.iloc[test_idx][['series_id', 'window', 'step']]
                       .groupby(['series_id', 'window'])
                       .apply(lambda x: x.iloc[0]))
        # FIXME This causes a crash later on in the compute_nan_confusion_matrix as it tries
        #  to access the first step as a negative index which is now a very large integer instead
        # get only the test series data from the solution
        validation_series_ids = window_info['series_id'].unique()
        # if visualize is true plot all test series
        with open('./series_id_encoding.json', 'r') as f:
            encoding = json.load(f)
        decoding = {v: k for k, v in encoding.items()}
        validation_series_ids = [decoding[sid] for sid in validation_series_ids]

        submission = to_submission_format(predictions, window_info)
        solution = (pd.read_csv(config.get_train_events_path())
                    .groupby('series_id')
                    .filter(lambda x: x['series_id'].iloc[0] in validation_series_ids)
                    .reset_index(drop=True))

        logger.info("Start scoring test predictions...")
        log_scores_to_wandb(compute_score_full(submission, solution), compute_score_clean(submission, solution))

        # compute confusion matrix for making predictions or not
        window_info['series_id'] = window_info['series_id'].map(decoding)
        compute_nan_confusion_matrix(submission, solution, window_info)

        # the plot function applies encoding to the submission
        # we do not want to change the ids on the original submission
        plot_submission = submission.copy()
        # pass only the test data
        logger.info('Creating plots...')
        plot_preds_on_series(plot_submission,
                             featured_data[
                                 featured_data['series_id'].isin(list(encoding[i] for i in validation_series_ids))],
                             number_of_series_to_plot=config.get_number_of_plots(),
                             folder_path='prediction_plots/' + config_hash,
                             show_plot=config.get_browser_plot(), save_figures=config.get_store_plots()),

    else:
        logger.info("Not scoring")

    # ------------------------------------------------------- #
    #                    Train for submission                 #
    # ------------------------------------------------------- #

    print_section_separator("Train for submission", spacing=0)

    if config.get_train_for_submission():
        logger.info("Retraining models for submission")

        # Retrain all models with optimal parameters
        X_train, y_train = pretrain.pretrain_final(featured_data)

        # Save scaler
        scaler_filename: str = config.get_model_store_loc() + "/scaler-" + initial_hash + ".pkl"
        pretrain.scaler.save(scaler_filename)

        for i, model in enumerate(models):
            model_filename_opt = store_location + "/optimal_" + model + "-" + initial_hash + models[model].hash + ".pt"
            model_filename_submit = store_location + "/submit_" + model + "-" + initial_hash + models[
                model].hash + ".pt"
            if os.path.isfile(model_filename_submit):
                logger.info("Found existing fully trained optimal model " + str(
                    i) + ": " + model + " with location " + model_filename)
            else:
                models[model].load(model_filename_opt, only_hyperparameters=True)
                logger.info("Retraining model " + str(i) + ": " + model)
                models[model].train_full(X_train, y_train)
                models[model].save(model_filename_submit)

    else:
        logger.info("Not training best model for submission")

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
