import gc
import os
import numpy as np

from src.configs.load_config import ConfigLoader
from src.get_processed_data import get_processed_data
from src.logger.logger import logger
from src.pretrain.pretrain import Pretrain
from src.util.hash_config import hash_config
from src.util.submissionformat import to_submission_format


def submit(config: ConfigLoader, submit=False) -> None:
    featured_data = get_processed_data(config, save_output=False, training=False)

    # Get predict with cpu
    pred_with_cpu = config.get_pred_with_cpu()

    # Hash of concatenated string of preprocessing, feature engineering and pretraining
    initial_hash = hash_config(config.get_pp_fe_pretrain(), length=5)

    # Apply pretraining
    pretrain: Pretrain = config.get_pretraining()
    if pretrain.scaler.scaler:
        pretrain.scaler.load(config.get_model_store_loc() + "/scaler-" + initial_hash + ".pkl")

    x_data = pretrain.preprocess(featured_data)

    gc.collect()

    # for the first step of each window get the series id and step offset
    important_cols = ['series_id', 'window', 'step'] + [col for col in featured_data.columns if 'similarity_nan' in col]
    grouped = (featured_data[important_cols]
               .groupby(['series_id', 'window']))
    window_offset = grouped.apply(lambda x: x.iloc[0])

    # filter out predictions using a threshold on (f_)similarity_nan
    filter_cfg = config.get_similarity_filter()
    nan_mask = None
    if filter_cfg:
        logger.info(f"Creating filter for predictions using similarity_nan with threshold: {filter_cfg['threshold']:.3f}")
        col_name = [col for col in featured_data.columns if 'similarity_nan' in col]
        if len(col_name) == 0:
            raise ValueError("No (f_)similarity_nan column found in the data for filtering")
        mean_sim = grouped.apply(lambda x: (x[col_name] == 0).mean())
        nan_mask = mean_sim > filter_cfg['threshold']
        nan_mask = np.where(nan_mask, np.nan, 1)

    # no longer need the full dataframe
    del featured_data
    gc.collect()

    # get models
    store_location = config.get_model_store_loc()
    models = config.get_models()
    for i, model in enumerate(models):
        model_filename_submit = store_location + "/submit_" + model + "-" + initial_hash + models[model].hash + ".pt"
        logger.info(f"Loading model {i}: {model} from {model_filename_submit}")
        models[model].load(model_filename_submit)

    # make predictions
    ensemble = config.get_ensemble(models)
    predictions = ensemble.pred(x_data, pred_with_cpu=pred_with_cpu)
    if nan_mask is not None:
        predictions = predictions * nan_mask

    formatted = to_submission_format(predictions, window_offset)

    if submit:
        formatted.to_csv("submission.csv")
        print(f"Saved submission.csv to {os.path.abspath('submission.csv')}")


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install()

    config = ConfigLoader("config.json")

    submit(config, submit=True)
