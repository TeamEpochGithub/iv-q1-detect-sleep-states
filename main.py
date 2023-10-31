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
from src.score.doscoring import compute_scores
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

    # ------------------------- #
    #         Pre-train         #
    # ------------------------- #

    print_section_separator("Pre-train", spacing=0)

    logger.info("Get pretraining parameters from config and initialize pretrain")
    pretrain: Pretrain = config.get_pretraining()

    logger.info("Pretraining with scaler " + str(pretrain.scaler.kind) + " and test size of " + str(pretrain.test_size))

    # Split data into train and test
    # Use numpy.reshape to turn the data into a 3D tensor with shape (window, n_timesteps, n_features)
    logger.info("Splitting data into train and test...")

    X_train, X_test, y_train, y_test, train_idx, test_idx = pretrain.pretrain_split(featured_data)

    # Give data shape in terms of (features (in_channels), window_size))
    data_shape = (X_train.shape[2], X_train.shape[1])

    logger.info("X Train data shape (size, window_size, features): " + str(
        X_train.shape) + " and y train data shape (size, window_size, features): " + str(y_train.shape))
    logger.info("X Test data shape (size, window_size, features): " + str(
        X_test.shape) + " and y test data shape (size, window_size, features): " + str(y_test.shape))

    # TODO Cross validation should be part of each model
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

    # Hash of concatenated string of preprocessing, feature engineering and pretraining
    initial_hash = hash_config(config.get_pp_fe_pretrain(), length=5)

    # TODO Add crossvalidation to models and hyperparameter optimization #107, #101
    for i, model in enumerate(models):
        # Get filename of model
        model_filename = store_location + "/" + model + "-" + initial_hash + models[model].hash + ".pt"
        # If this file exists, load instead of start training
        if os.path.isfile(model_filename):
            logger.info("Found existing trained model " + str(i) + ": " + model + " with location " + model_filename)
            models[model].load(model_filename, only_hyperparameters=False)
        else:
            logger.info("Training model " + str(i) + ": " + model)
            models[model].train(X_train, X_test, y_train, y_test)
            models[model].save(model_filename)
            logger.info("Training model " + str(i) + ": " + model)

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
        predictions = ensemble.pred(X_test, pred_cpu)

        logger.info("Formatting predictions...")

        # TODO simplify this
        # for each window get the series id and step offset
        important_cols = ['series_id', 'window', 'step'] + [col for col in featured_data.columns if 'similarity_nan' in col]
        grouped = (featured_data.iloc[test_idx][important_cols]
                   .groupby(['series_id', 'window']))
        window_offset = grouped.apply(lambda x: x.iloc[0])

        # filter out predictions using a threshold on (f_)similarity_nan
        filter_cfg = config.get_similarity_filter()
        if filter_cfg:
            logger.info(f"Filtering predictions using similarity_nan with threshold: {filter_cfg['threshold']:.3f}")
            col_name = [col for col in featured_data.columns if 'similarity_nan' in col]
            if len(col_name) == 0:
                raise ValueError("No (f_)similarity_nan column found in the data for filtering")
            mean_sim = grouped.apply(lambda x: (x[col_name] == 0).mean())
            nan_mask = mean_sim > filter_cfg['threshold']
            nan_mask = np.where(nan_mask, np.nan, 1)
            predictions = predictions * nan_mask

        submission = to_submission_format(predictions, window_offset)

        # get only the test series data from the solution
        test_series_ids = window_offset['series_id'].unique()

        # if visualize is true plot all test series
        with open('./series_id_encoding.json', 'r') as f:
            encoding = json.load(f)
        decoding = {v: k for k, v in encoding.items()}
        test_series_ids = [decoding[sid] for sid in test_series_ids]

        # load solution for test set and compute score
        solution = (pd.read_csv(config.get_train_events_path())
                    .groupby('series_id')
                    .filter(lambda x: x['series_id'].iloc[0] in test_series_ids)
                    .reset_index(drop=True))
        logger.info("Start scoring test predictions...")
        result = compute_scores(submission, solution)

        # compute confusion matrix for making predictions or not
        window_offset['series_id'] = window_offset['series_id'].map(decoding)
        compute_nan_confusion_matrix(submission, solution, window_offset)

        # the plot function applies encoding to the submission
        # we do not want to change the ids on the original submission
        plot_submission = submission.copy()

        # pass only the test data
        logger.info('Creating plots...')
        plot_preds_on_series(plot_submission,
                             featured_data[featured_data['series_id'].isin(list(encoding[i] for i in test_series_ids))],
                             number_of_series_to_plot=config.get_number_of_plots(),
                             folder_path='prediction_plots/' + config_hash + f'-Score--{result:.4f}',
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
                    i) + ": " + model + " with location " + model_filename_submit)
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
