import gc
import os

from src.configs.load_config import ConfigLoader
from src.get_processed_data import get_processed_data
from src.logger.logger import logger
from src.pre_train.standardization import standardize
from src.util.submissionformat import to_submission_format


def submit(config: ConfigLoader, submit=False) -> None:
    featured_data = get_processed_data(config, save_output=False, training=False)

    # format the data
    feature_cols = [col for col in featured_data.columns if col.startswith('f_')]
    x_data = featured_data[['enmo', 'anglez'] + feature_cols]

    # apply pretraining
    pretrain = config.get_pretraining()
    x_data = standardize(x_data, method=pretrain["standardize"])
    x_data = x_data.to_numpy(dtype='float32').reshape(-1, 17280, len(x_data.columns))
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
        path = store_location + "/submit_" + model + ".pt"
        logger.info(f"Loading model {i}: {model} from {path}")
        config.models[model].load(path)

    # make predictions
    ensemble = config.get_ensemble(models)
    predictions = ensemble.pred(x_data)

    formatted = to_submission_format(predictions, window_info)

    if submit:
        formatted.to_csv("submission.csv")
        print(f"Saved submission.csv to {os.path.abspath('submission.csv')}")


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install()

    config = ConfigLoader("config.json")

    submit(config, submit=True)

