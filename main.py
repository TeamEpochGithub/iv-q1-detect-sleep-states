import os
import random

import numpy as np
import torch

import wandb
from main_utils import train_from_config, full_train_from_config, scoring
from src import data_info
from src.configs.load_config import ConfigLoader
from src.logger.logger import logger
from src.util.hash_config import hash_config
from src.util.printing_utils import print_section_separator
from sweep import play_mp3


def main() -> None:
    """
    Main function for training the model
    :param config: loaded config
    """
    print_section_separator("Q1 - Detect Sleep States - Kaggle", spacing=0)
    logger.info("Start of main.py")

    # Load config file and hash
    global config_loader
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
        # Add models
        is_hpo = config_loader.get_hpo()
        is_ensemble_hpo = config_loader.get_ensemble_hpo()

        if is_hpo and is_ensemble_hpo:
            logger.critical("Cannot have both hpo and ensemble hpo")
            raise Exception("Cannot have both hpo and ensemble hpo")

        if is_ensemble_hpo:
            ensemble_config = is_ensemble_hpo
            min_models, max_models = (2, 6)
            n_models = np.random.randint(min_models, max_models)

            ensemble_models = os.listdir(ensemble_config["model_config_loc"])
            chosen_models = random.choices(ensemble_models, k=n_models)
            chosen_weights = [random.random() for _ in range(n_models)]

            # Override ensemble models with the chosen models
            config_loader.config["ensemble"]["models"] = chosen_models
            config_loader.config["ensemble"]["weights"] = chosen_weights

            # Set the ensemble config
            config_loader.set_ensemble()

            # Merge the wandb config and the ensemble config
            logger.info("ENSEMBLE HPO BEFORE" + str(config_loader.config["ensemble_hpo"]))
            config_loader.config["ensemble_hpo"] |= wandb.config.get("ensemble_hpo")
            logger.info("ENSEMBLE HPO AFTER" + str(config_loader.config["ensemble_hpo"]))

        if is_hpo:
            # Get the hpo config and add it to the config on wandb
            config_loader.config["hpo_model"] = config_loader.get_hpo_config().config

            # Merge the config from the hpo config
            config_loader.config |= wandb.config
            assert config_loader.config["hpo_model"] == wandb.config["hpo_model"]

            # Update the hpo config with the merged config
            config_loader.get_hpo_config().config = config_loader.config.get("hpo_model")

            # Update hash as the config is different now
            config_hash = hash_config(config_loader.get_config(), length=16)

            wandb.run.name = config_hash

            # Update the wandb summary with the updated config
            curr_config = config_loader.get_config()
        else:
            # Get the ensemble configs and add them to the config on wandb
            ensemble = config_loader.get_ensemble()
            models = ensemble.get_models()
            for i, model_config in enumerate(models):
                wandb.config[f"model_{i}"] = model_config.get_config()

            # Update the wandb summary also with the ensemble configs
            curr_config = config_loader.get_config()
            model_names = config_loader.get_config()["ensemble"]["models"]

            for name, model in zip(model_names, models):
                curr_config[name] = model.config

        # Update the wandb summary with the updated config
        wandb.run.summary.update(curr_config)
        logger.info(f"Logging to wandb with run id: {config_hash}")
    else:
        logger.info("Not logging to wandb")

        if config_loader.get_hpo():
            logger.critical("HPO requires wandb")
            raise Exception("HPO requires wandb")

    # Predict with CPU
    pred_cpu = config_loader.get_pred_with_cpu()
    if pred_cpu:
        logger.info("Predicting with CPU for inference")
    else:
        logger.info("Predicting with GPU for inference")

    # Store location
    store_location = config_loader.get_model_store_loc()
    logger.info("Model store location: " + store_location)

    if not config_loader.get_train_for_submission():
        # ------------------------------------------- #
        #                 Ensemble                    #
        # ------------------------------------------- #

        # If hpo is enabled, run hpo instead
        # This can be done via terminal and a sweep or a local cross validation run
        if config_loader.get_hpo():
            logger.info("Running HPO")
            train_from_config(config_loader.get_hpo_config(),
                              config_loader.get_cv(), store_location, hpo=True)
            return

        # Initialize models
        logger.info("Initializing models...")
        ensemble = config_loader.get_ensemble()
        models = ensemble.get_models()
        if not ensemble.get_pred_only():
            for _, model_config in enumerate(models):
                train_from_config(model_config, config_loader.get_cv(),
                                  store_location, hpo=False)
        else:
            logger.info("Not training models")

        # ------------------------------------------------------- #
        #                    Scoring                              #
        # ------------------------------------------------------- #

        print_section_separator("Scoring", spacing=0)
        data_info.stage = "scoring"
        data_info.substage = ""

        if config_loader.get_scoring():
            scoring(config=config_loader)
        else:
            logger.info("Not scoring")
    else:
        # ------------------------------------------------------- #
        #                    Train for submission                 #
        # ------------------------------------------------------- #

        print_section_separator("Train for submission", spacing=0)
        data_info.stage = "train for submission"
        data_info.substage = "Full"
        for model_config in ensemble.get_models():
            full_train_from_config(model_config, store_location)

    # [optional] finish the wandb run, necessary in notebooks
    if config_loader.get_log_to_wandb():
        wandb.finish()
        logger.info("Finished logging to wandb")


if __name__ == "__main__":

    # Set up logging
    import coloredlogs

    coloredlogs.install()

    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(42)
    # Load config file
    config_loader: ConfigLoader = ConfigLoader("config.json")

    # Gotta sweep
    if config_loader.get_hpo():
        mp3_file_path = "gotta_sweep.mp3"  # Replace with the path to your MP3 file
        play_mp3(mp3_file_path)
        # try:
        main()
        # except Exception as e:
        #     if config_loader.get_log_to_wandb():
        #         wandb.log({"cv_score": -0.1})
        #         wandb.log({"exception": str(e)})
    else:
        main()
