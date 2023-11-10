# This file does the training of the model
import gc
import json
import os

import numpy as np
import pandas as pd

import wandb
from src import data_info
from src.configs.load_config import ConfigLoader
from src.configs.load_model_config import ModelConfigLoader
from src.ensemble.ensemble import Ensemble
from src.get_processed_data import get_processed_data
from src.logger.logger import logger
from src.pretrain.pretrain import Pretrain
from src.score.compute_score import log_scores_to_wandb, compute_score_full, compute_score_clean
from src.score.nan_confusion import compute_nan_confusion_matrix
from src.score.visualize_preds import plot_preds_on_series
from src.util.hash_config import hash_config
from src.util.printing_utils import print_section_separator
from src.util.submissionformat import to_submission_format
from main_utils import train_from_config, full_train_from_config, scoring


def main() -> None:
    """
    Main function for training the model
    :param config: loaded config
    """
    print_section_separator("Q1 - Detect Sleep States - Kaggle", spacing=0)
    logger.info("Start of main.py")

    global config_loader
    config_loader.reset_globals()
    config_hash = hash_config(config_loader.get_config(), length=16)
    logger.info("Config hash encoding: " + config_hash)

    # Initialize wandb
    if config_loader.get_log_to_wandb():
        # Initialize wandb
        wandb.init(
            project='detect-sleep-states',
            name=config_hash,
            config=config_loader.get_config()
        )
        if config_loader.get_hpo():
            config_loader.config |= wandb.config

        wandb.run.summary.update(config_loader.get_config())
        logger.info(f"Logging to wandb with run id: {config_hash}")
    else:
        logger.info("Not logging to wandb")

    # Predict with CPU
    pred_cpu = config_loader.get_pred_with_cpu()
    if pred_cpu:
        logger.info("Predicting with CPU for inference")
    else:
        logger.info("Predicting with GPU for inference")

    # ------------------------------------------- #
    #                 Ensemble                    #
    # ------------------------------------------- #

    # Initialize models
    store_location = config_loader.get_model_store_loc()
    logger.info("Model store location: " + store_location)

    # Initialize models
    logger.info("Initializing models...")

    ensemble = config_loader.get_ensemble()
    models = ensemble.get_models()
    if not ensemble.get_pred_only():
        for _, model_config in enumerate(models):
            train_from_config(model_config, config_loader, store_location)
    else:
        logger.info("Not training models")

    # ------------------------------------------------------- #
    #                    Scoring                              #
    # ------------------------------------------------------- #

    print_section_separator("Scoring", spacing=0)
    data_info.stage = "scoring"
    data_info.substage = ""

    if config_loader.get_scoring():
        scoring(config=config_loader, ensemble=ensemble)
    else:
        logger.info("Not scoring")

    # ------------------------------------------------------- #
    #                    Train for submission                 #
    # ------------------------------------------------------- #

    print_section_separator("Train for submission", spacing=0)
    data_info.stage = "train for submission"

    if config_loader.get_train_for_submission():
        data_info.substage = "Full"
        for model_config in ensemble.get_models():
            config_loader.reset_globals()
            full_train_from_config(model_config, store_location)
        logger.info("Retraining models for submission")
    else:
        logger.info("Not training best model for submission")

    # [optional] finish the wandb run, necessary in notebooks
    if config_loader.get_log_to_wandb():
        wandb.finish()
        logger.info("Finished logging to wandb")


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install()

    # Load config file
    config_loader: ConfigLoader = ConfigLoader("config.json")
    main()
