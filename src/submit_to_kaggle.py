import os

from src import data_info
from src.configs.load_config import ConfigLoader
from src.logger.logger import logger
from src.util.submissionformat import to_submission_format


def submit(config_loader: ConfigLoader, submit=False) -> None:
    logger.info("Making predictions with ensemble on kaggle data")

    # Check if data/processed exists, if not create it
    processed_path = config_loader.get_processed_out()
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)

    # Predict with CPU
    pred_cpu = config_loader.get_pred_with_cpu()
    if pred_cpu:
        logger.info("Predicting with CPU for inference")
    else:
        logger.info("Predicting with GPU for inference")

    # Get ensemble
    ensemble = config_loader.get_ensemble()
    if not ensemble:
        raise ValueError("No ensemble found in config")

    is_kaggle = config_loader.config.get("is_kaggle", False)
    # Make predictions on test data
    predictions = ensemble.pred(config_loader.get_model_store_loc(), pred_with_cpu=pred_cpu, training=False,
                                is_kaggle=is_kaggle)

    logger.info("Formatting predictions...")

    submission = to_submission_format(predictions, data_info.window_info)

    if submit:
        submission.to_csv("submission.csv")
        print(f"Saved submission.csv to {os.path.abspath('submission.csv')}")


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install()

    config_loader = ConfigLoader("config.json")

    submit(config_loader, submit=True)
