import os

from src.get_processed_data import get_processed_data
from src.configs.load_config import ConfigLoader
from src.util.submissionformat import to_submission_format


def submit(config: ConfigLoader, test_series_path, submit=False):

    featured_data = get_processed_data(config, test_series_path, save_output=False)
    # Initialize models
    models = config.get_models()

    # Get saved models directory from config
    store_location = config.get_model_store_loc()

    # load models
    for model in config.models:
        config.models[model].load(store_location)

    ensemble = config.get_ensemble(models)

    predictions = ensemble.pred(featured_data)

    formatted = to_submission_format(predictions)

    if submit:
        formatted.to_csv("submission.csv")
        print(f"Saved submission.csv to {os.path.abspath('submission.csv')}")


if __name__ == "__main__":
    config = ConfigLoader("config.json")
    submit(config, 'data/raw/test_series.parquet', submit=True)
