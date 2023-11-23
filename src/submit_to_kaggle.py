import os
import numpy as np

from src.configs.load_config import ConfigLoader
from src.get_processed_data import get_processed_data
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
    predictions = ensemble.pred(config_loader.get_model_store_loc(), pred_with_cpu=pred_cpu, training=False, is_kaggle=is_kaggle)

    # Get featured data for model 1, should not give any problems as all models should have the same columns excluding features
    ensemble.get_models()[0].reset_globals()
    featured_data = get_processed_data(
        ensemble.get_models()[0], training=False, save_output=not is_kaggle)

    logger.info("Formatting predictions...")

    # for the first step of each window get the series id and step offset
    important_cols = ['series_id', 'window', 'step'] + [col for col in featured_data.columns if 'similarity_nan' in col]
    grouped = (featured_data[important_cols]
               .groupby(['series_id', 'window']))
    window_offset = grouped.apply(lambda x: x.iloc[0])

    # filter out predictions using a threshold on (f_)similarity_nan
    filter_cfg = config_loader.get_similarity_filter()
    if filter_cfg:
        logger.info(f"Creating filter for predictions using similarity_nan with threshold: {filter_cfg['threshold']:.3f}")
        col_name = [col for col in featured_data.columns if 'similarity_nan' in col]
        if len(col_name) == 0:
            raise ValueError("No (f_)similarity_nan column found in the data for filtering")
        mean_sim = grouped.apply(lambda x: (x[col_name] == 0).mean())
        nan_mask = mean_sim > filter_cfg['threshold']
        nan_mask = np.where(nan_mask, np.nan, 1)
        predictions = predictions * nan_mask

    submission = to_submission_format(predictions, window_offset)

    if submit:
        submission.to_csv("submission.csv")
        print(f"Saved submission.csv to {os.path.abspath('submission.csv')}")


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install()

    config_loader = ConfigLoader("config.json")

    submit(config_loader, submit=True)
