import os

import numpy as np
import pandas as pd

from src import data_info
from src.configs.load_config import ConfigLoader
from src.configs.load_model_config import ModelConfigLoader
from src.cv.cv import CV
from src.get_processed_data import get_processed_data
from src.logger.logger import logger
from src.pretrain.pretrain import Pretrain
from src.score.compute_score import compute_score_full, compute_score_clean, log_scores_to_wandb
from src.score.nan_confusion import compute_nan_confusion_matrix
from src.util.get_pretrain_cache import get_pretrain_full_cache, get_pretrain_split_cache
from src.util.hash_config import hash_config
from src.util.printing_utils import print_section_separator
from src.util.submissionformat import to_submission_format


def train_from_config(model_config: ModelConfigLoader, cross_validation: CV, store_location: str, hpo: bool = False) -> None:

    # Initialisation
    model_config.reset_globals()
    model_name = model_config.get_name()

    # ------------------------------------------- #
    #    Preprocessing and feature Engineering    #
    # ------------------------------------------- #

    print_section_separator(
        "Preprocessing and feature engineering", spacing=0)
    data_info.stage = "preprocessing & feature engineering"
    featured_data = get_processed_data(
        model_config, training=True, save_output=True)

    # ------------------------ #
    #         Pretrain         #
    # ------------------------ #

    print_section_separator("Pretrain", spacing=0)
    data_info.stage = "pretraining"

    logger.info(
        "Get pretraining parameters from config and initialize pretrain")
    pretrain: Pretrain = model_config.get_pretraining()

    logger.info("Pretraining with scaler " + str(pretrain.scaler.kind) +
                " and test size of " + str(pretrain.test_size))

    # Split data into train/test and validation
    # Use numpy.reshape to turn the data into a 3D tensor with shape (window, n_timesteps, n_features)
    logger.info("Splitting data into train and test...")
    data_info.substage = "pretrain_split"

    x_train, x_test, y_train, y_test, train_ids, _, groups = get_pretrain_split_cache(
        model_config, featured_data, save_output=True)

    logger.info("X Train data shape (size, window_size, features): " + str(
        x_train.shape) + " and y Train data shape (size, window_size, features): " + str(y_train.shape))
    logger.info("X Test data shape (size, window_size, features): " + str(
        x_test.shape) + " and y Test data shape (size, window_size, features): " + str(y_test.shape))
    logger.info("Creating model using ModelConfigLoader")

    assert x_test.shape[1] == data_info.window_size_before // data_info.downsampling_factor == data_info.window_size == y_test.shape[1] == x_train.shape[1] == y_train.shape[1]

    model = model_config.set_model()

    # Hash of concatenated string of preprocessing, feature engineering and pretraining
    initial_hash = hash_config(
        model_config.get_pp_fe_pretrain(), length=5)
    data_info.substage = f"training model: {model_name}"

    # Get filename of model
    model_filename_opt = store_location + "/optimal_" + \
        model_name + "-" + initial_hash + model.hash + ".pt"

    # Get cv object
    cv = cross_validation

    def run_cv():

        # ------------------------- #
        # Cross Validation Training #
        # ------------------------- #

        print_section_separator("CV", spacing=0)
        logger.info("Applying cross-validation on model: " + model_name)
        data_info.stage = "cv"

        # It now only saves the trained model from the last fold
        train_window_info = data_info.window_info[data_info.window_info['series_id'].isin(train_ids)]

        # Apply CV
        scores = cv.cross_validate(
            model, x_train, y_train, train_window_info=train_window_info, groups=groups)

        # Log scores to wandb
        mean_scores = np.mean(scores, axis=0)
        log_scores_to_wandb(mean_scores, data_info.scorings)
        logger.info(
            f"Done CV for model: {model_name} with CV scores of \n {scores} and mean score of {np.mean(scores, axis=0)}")

    # If this file exists, load instead of start training
    if hpo:
        run_cv()
        return
    else:
        data_info.stage = "train"
        data_info.substage = "optimal"
        if os.path.isfile(model_filename_opt):
            logger.info("Found existing trained optimal model: " +
                        model_name + " with location " + model_filename_opt)
            model.load(model_filename_opt, only_hyperparameters=False)
        else:
            logger.info("Training optimal model: " + model_name)
            model.train(x_train, x_test, y_train, y_test)

    model.save(model_filename_opt)


def scoring(config: ConfigLoader) -> None:
    logger.info("Making predictions with ensemble on test data")
    # Predict with CPU
    pred_cpu = config.get_pred_with_cpu()
    if pred_cpu:
        logger.info("Predicting with CPU for inference")
    else:
        logger.info("Predicting with GPU for inference")

    # Get ensemble
    ensemble = config.get_ensemble()

    # Make predictions on test data
    predictions = ensemble.pred(
        config.get_model_store_loc(), pred_with_cpu=pred_cpu)
    test_ids = ensemble.get_test_ids()

    logger.info("Formatting predictions...")

    test_window_info = data_info.window_info[data_info.window_info['series_id'].isin(test_ids)]
    submission = to_submission_format(predictions, test_window_info)

    # load solution for test set and compute score
    solution = (pd.read_csv(config.get_train_events_path())
                .groupby('series_id')
                .filter(lambda x: x['series_id'].iloc[0] in test_ids)
                .reset_index(drop=True))
    logger.info("Start scoring test predictions...")

    scores = [compute_score_full(
        submission, solution), compute_score_clean(submission, solution)]
    log_scores_to_wandb(scores, data_info.scorings)

    # compute confusion matrix for making predictions or not
    compute_nan_confusion_matrix(submission, solution, data_info.window_info)

    # pass only the test data
    logger.info('Creating plots...')

    # TODO make plots work again with ensemble, which featured data to use?
    # plot_preds_on_series(submission,
    #                      featured_data[
    #                          featured_data['series_id'].isin(test_ids)],
    #                      number_of_series_to_plot=config.get_number_of_plots(),
    #                      folder_path=f'prediction_plots/{config_hash}-Score--{scores[1]:.4f}',
    #                      show_plot=config.get_browser_plot(), save_figures=config.get_store_plots())


def full_train_from_config(model_config_loader: ModelConfigLoader, store_location: str) -> None:
    """
    Full train the model with the optimal parameters
    :param model_config_loader: the model config
    :param store_location: the store location of the models
    """
    # Initialisation
    model_config_loader.reset_globals()
    model_name = model_config_loader.get_name()

    initial_hash = hash_config(
        model_config_loader.get_pretrain_config(), length=5)

    featured_data = get_processed_data(
        model_config_loader, training=True, save_output=True)

    x_train, y_train = get_pretrain_full_cache(
        model_config_loader, featured_data, save_output=True)

    logger.info("Creating model using ModelConfigLoader")
    model = model_config_loader.set_model()

    model_filename_opt = store_location + "/optimal_" + \
        model_name + "-" + initial_hash + model.hash + ".pt"
    model_filename_submit = store_location + "/submit_" + \
        model_name + "-" + initial_hash + model.hash + ".pt"
    if os.path.isfile(model_filename_submit):
        logger.info("Found existing fully trained submit model: " +
                    model_name + " with location " + model_filename_submit)
    elif os.path.isfile(model_filename_opt):
        model.load(model_filename_opt, only_hyperparameters=True)
        logger.info("Training fully trained submit model: " + model_name)
        model.train_full(x_train, y_train)
        model.save(model_filename_submit)
    else:
        raise ValueError(
            f"Could not find optimal model: {model_name} with location {model_filename_opt}." +
            " Please train the optimal model first by setting train_for_submission to false and then try again.")
