import gc
import os

from src.configs.load_config import ConfigLoader
from src.get_processed_data import get_processed_data
from src.logger.logger import logger
from src.pretrain.pretrain import Pretrain
from src.util.hash_config import hash_config
from src.util.submissionformat import to_submission_format


def submit(config: ConfigLoader, submit=False) -> None:
    featured_data = get_processed_data(config, save_output=False, training=False)

    # Get predict with cpu
    pred_cpu = config.get_pred_with_cpu()

    # Hash of concatenated string of preprocessing, feature engineering and pretraining
    initial_hash = hash_config(config.get_pp_fe_pretrain(), length=5)

    # Apply pretraining
    pretrain: Pretrain = config.get_pretraining()
    pretrain.scaler.load(config.get_model_store_loc() + "/scaler-" + initial_hash + ".pkl")

    x_data = pretrain.preprocess(featured_data)

    gc.collect()

    # for the first step of each window get the series id and step offset
    window_info = (featured_data[['series_id', 'window', 'step']]
                   .groupby(['series_id', 'window'])
                   .apply(lambda x: x.iloc[0]))

    # no longer need the full dataframe
    del featured_data
    gc.collect()

    # get models
    store_location = config.get_model_store_loc()
    data_shape = (x_data.shape[2], x_data.shape[1])
    models = config.get_models(data_shape)
    for i, model in enumerate(models):
        model_filename_submit = store_location + "/submit_" + model + "-" + initial_hash + models[model].hash + ".pt"
        logger.info(f"Loading model {i}: {model} from {model_filename_submit}")
        models[model].load(model_filename_submit)

    # make predictions
    ensemble = config.get_ensemble(models)
    predictions = ensemble.pred(x_data, pred_cpu=pred_cpu)

    formatted = to_submission_format(predictions, window_info)

    if submit:
        formatted.to_csv("submission.csv")
        print(f"Saved submission.csv to {os.path.abspath('submission.csv')}")


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install()

    config = ConfigLoader("config.json")

    submit(config, submit=True)
