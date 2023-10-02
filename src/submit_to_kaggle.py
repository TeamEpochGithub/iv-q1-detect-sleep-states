import os

import pandas as pd

from src.configs.load_config import ConfigLoader
from src.util.submissionformat import to_submission_format


def submit(config: ConfigLoader, test_series_path, submit=False):
    df = pd.read_parquet(test_series_path)

    # Initialize preprocessing steps
    pp_steps, pp_s = config.get_pp_steps(training=False)
    processed = df
    # Get the preprocessing steps as a list of str to make the paths
    for i, step in enumerate(pp_steps):
        # Passes the current list because it's needed to write to if the path doesn't exist
        processed = step.run(processed, pp_s[:i + 1], save_result=False)

    # Initialize feature engineering steps
    fe_steps, fe_s = config.get_features()
    featured_data = processed
    for i, fe_step in enumerate(fe_steps):
        # Also pass the preprocessing steps to the feature engineering step
        # to save fe for each possible pp combination
        featured_data = fe_steps[fe_step].run(processed, fe_s[:i + 1], pp_s, save_result=False)

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
