# This file does the training of the model

# Imports
import wandb

from src import submit_to_kaggle
from src.configs.load_config import ConfigLoader
from src.logger.logger import logger
from src.util.printing_utils import print_section_separator
from src.pre_train.train_test_split import train_test_split
from src.get_processed_data import get_processed_data


def main(config: ConfigLoader, series_path) -> None:
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

    featured_data = get_processed_data(config, series_path, save_output=True)

    # ------------------------- #
    #         Pre-train         #
    # ------------------------- #

    print_section_separator("Pre-train", spacing=0)

    pretrain = config.get_pretraining()

    # Use numpy.reshape to turn the data into a 3D tensor with shape (window, n_timesteps, n_features)
    X_train, X_test, Y_train, Y_test = train_test_split(featured_data, test_size=pretrain["test_size"], standardize_method=pretrain["standardize"])

    cv = 0
    if "cv" in pretrain:
        cv = config.get_cv()

    # ------------------------- #
    #          Training         #
    # ------------------------- #

    print_section_separator("Training", spacing=0)

    # Initialize models
    print("-------- TRAINING MODELS ----------")
    models = config.get_models()
    store_location = config.get_model_store_loc()
    logger.info("Model store location: " + store_location)

    # TODO Add crossvalidation to models #107
    for i, model in enumerate(models):
        logger.info("Training model " + str(i) + ": " + model)
        models[model].train(X_train, X_test, Y_train, Y_test)
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
    test_data = None
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
    import coloredlogs
    coloredlogs.install()

    # Load config file
    config = ConfigLoader("config.json")

    # Run main
    main(config, "data/raw/test_series.parquet")

    # Create submission
    submit_to_kaggle.submit(config, "data/raw/test_series.parquet", False)
