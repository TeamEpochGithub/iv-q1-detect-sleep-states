# This file does the training of the model
import json
import os

import numpy as np
import pandas as pd

import wandb
from src import data_info
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
        logger.info("Predicting with CPU for inference")
    else:
        logger.info("Predicting with GPU for inference")

    # ------------------------------------------- #
    #                 Ensemble                    #
    # ------------------------------------------- #

    # Initialize models
    store_location = config.get_model_store_loc()
    logger.info("Model store location: " + store_location)

    # Initialize models
    logger.info("Initializing models...")

    ensemble = config.get_ensemble()
    models = ensemble.get_models()
    if not ensemble.get_pred_only():
        for i, model_config in enumerate(models):
            config.reset_globals()
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

            x_train, x_test, y_train, y_test, train_idx, test_idx, groups = pretrain.pretrain_split(
                featured_data)

            logger.info("X Train data shape (size, window_size, features): " + str(
                x_train.shape) + " and y Train data shape (size, window_size, features): " + str(y_train.shape))
            logger.info("X Test data shape (size, window_size, features): " + str(
                x_test.shape) + " and y Test data shape (size, window_size, features): " + str(y_test.shape))

            logger.info("Creating model using ModelConfigLoader")
            model = model_config.set_model()

            # Hash of concatenated string of preprocessing, feature engineering and pretraining
            initial_hash = hash_config(
                model_config.get_pp_fe_pretrain(), length=5)
            data_info.substage = f"training model {i}: {model_name}"

            # Get filename of model
            model_filename_opt = store_location + "/optimal_" + \
                model_name + "-" + initial_hash + model.hash + ".pt"

            # Get cv object
            cv = config.get_cv()

            def run_cv():
                # ------------------------- #
                # Cross Validation Training #
                # ------------------------- #

                print_section_separator("CV", spacing=0)
                logger.info("Applying cross-validation on model " +
                            str(i) + ": " + model_name)
                data_info.stage = "cv"

                # TODO Implement hyperparameter optimization and train optimal model on train split and evaluate on test split. Save that as the optimal model.#101
                # It now only saves the trained model from the last fold
                train_df = featured_data.iloc[train_idx]

                # Apply CV
                scores = cv.cross_validate(
                    model, x_train, y_train, train_df=train_df, groups=groups)

                # Log scores to wandb
                mean_scores = np.mean(scores, axis=0)
                log_scores_to_wandb(mean_scores[0], mean_scores[1])
                logger.info(
                    f"Done CV for model {i}: {model} with CV scores of \n {scores} and mean score of {np.round(np.mean(scores, axis=0), 4)}")

            # If this file exists, load instead of start training
            if os.path.isfile(model_filename_opt):
                logger.info("Found existing trained optimal model " + str(i) +
                            ": " + model_name + " with location " + model_filename_opt)
                model.load(model_filename_opt, only_hyperparameters=False)
            else:
                if cv.get_apply():
                    run_cv()
                if config.get_train_optimal():
                    data_info.stage = "train"
                    data_info.substage = "optimal"

                    logger.info("Training optimal model " +
                                str(i) + ": " + model_name)
                    model.train(x_train, x_test, y_train, y_test)
                else:
                    logger.info("Not training optimal model " +
                                str(i) + ": " + model_name)
                    # Exit from main as the model is not trained optimally
                    if config.get_log_to_wandb():
                        wandb.finish()
                        logger.info("Finished logging to wandb")
                    return

            model.save(model_filename_opt)
    else:
        logger.info("Not training models")

    # ------------------------------------------------------- #
    #                    Scoring                              #
    # ------------------------------------------------------- #

    print_section_separator("Scoring", spacing=0)
    data_info.stage = "scoring"
    data_info.substage = ""

    scoring = config.get_scoring()
    if scoring:
        logger.info("Making predictions with ensemble on test data")
        predictions = ensemble.pred(config, pred_with_cpu=pred_cpu)

        logger.info("Formatting predictions...")

        # TODO simplify this
        # for each window get the series id and step offset
        # window_info = (featured_data.iloc[test_idx][['series_id', 'window', 'step']]
        #                .groupby(['series_id', 'window'])
        #                .apply(lambda x: x.iloc[0]))
        # # FIXME This causes a crash later on in the compute_nan_confusion_matrix as it tries
        # #  to access the first step as a negative index which is now a very large integer instead
        important_cols = ['series_id', 'window', 'step'] + \
            [col for col in featured_data.columns if 'similarity_nan' in col]
        grouped = (featured_data.iloc[test_idx][important_cols]
                   .groupby(['series_id', 'window']))
        window_offset = grouped.apply(lambda x: x.iloc[0])
        # TODO Check if the large step is still a problem now that we use the window_offset
        # TODO Do the window_offset thing in from_numpy_to_submission_format too

        # filter out predictions using a threshold on (f_)similarity_nan
        filter_cfg = config.get_similarity_filter()
        if filter_cfg:
            logger.info(
                f"Filtering predictions using similarity_nan with threshold: {filter_cfg['threshold']:.3f}")
            col_name = [
                col for col in featured_data.columns if 'similarity_nan' in col]
            if len(col_name) == 0:
                raise ValueError(
                    "No (f_)similarity_nan column found in the data for filtering")
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

        scores = (compute_score_full(submission, solution),
                  compute_score_clean(submission, solution))
        log_scores_to_wandb(*scores)

        # compute confusion matrix for making predictions or not
        window_offset['series_id'] = window_offset['series_id'].map(decoding)
        compute_nan_confusion_matrix(submission, solution, window_offset)

        # the plot function applies encoding to the submission
        # we do not want to change the ids on the original submission
        plot_submission = submission.copy()

        # pass only the test data
        logger.info('Creating plots...')
        plot_preds_on_series(plot_submission,
                             featured_data[
                                 featured_data['series_id'].isin(list(encoding[i] for i in test_series_ids))],
                             number_of_series_to_plot=config.get_number_of_plots(),
                             folder_path=f'prediction_plots/{config_hash}-Score--{scores[1]:.4f}',
                             show_plot=config.get_browser_plot(), save_figures=config.get_store_plots()),

    else:
        logger.info("Not scoring")

    # ------------------------------------------------------- #
    #                    Train for submission                 #
    # ------------------------------------------------------- #

    print_section_separator("Train for submission", spacing=0)
    data_info.stage = "train for submission"

    if config.get_train_for_submission():
        logger.info("Retraining models for submission")

        # Retrain all models with optimal parameters
        x_train, y_train, groups = pretrain.pretrain_final(featured_data)

        # Save scaler
        scaler_filename: str = config.get_model_store_loc() + "/scaler-" + \
            initial_hash + ".pkl"
        pretrain.scaler.save(scaler_filename)

        for i, model in enumerate(models):
            data_info.substage = "Full"

            model_filename_opt = store_location + "/optimal_" + \
                model + "-" + initial_hash + models[model].hash + ".pt"
            model_filename_submit = store_location + "/submit_" + model + "-" + initial_hash + models[
                model].hash + ".pt"
            if os.path.isfile(model_filename_submit):
                logger.info("Found existing fully trained submit model " + str(
                    i) + ": " + model + " with location " + model_filename_submit)
            else:
                models[model].load(model_filename_opt,
                                   only_hyperparameters=True)
                logger.info("Retraining model " + str(i) + ": " + model)
                models[model].train_full(x_train, y_train)
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
