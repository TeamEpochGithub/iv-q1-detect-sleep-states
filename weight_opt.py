import numpy as np
import optuna
import pandas as pd
import torch
from tqdm import tqdm

from src import data_info
from src.configs.load_config import ConfigLoader
from src.logger.logger import logger
from src.score.compute_score import compute_score_full
from src.util.hash_config import hash_config
from src.util.state_to_event import pred_to_event_state
from src.util.submissionformat import to_submission_format


def pred_all(ensemble, store_location):
    """Make predictions for all models and return all event confidences"""

    # Run each model
    predictions = None

    # model_pred is (onset, wakeup) tuples for each window
    for _, model_config in enumerate(ensemble.model_configs):
        model_config.reset_globals()
        model_pred = ensemble.pred_model(
            model_config_loader=model_config, store_location=store_location, pred_with_cpu=False,
            training=True, is_kaggle=False)

        # Model_pred is tuple of np.array(onset, awake) for each window
        # Split the series of tuples into two column
        if predictions is not None:
            predictions = np.concatenate((predictions, (model_pred.reshape(
                model_pred.shape[0], model_pred.shape[1], 2, 1))), axis=3)
        else:
            predictions = model_pred.reshape(
                model_pred.shape[0], model_pred.shape[1], 2, 1)

    return predictions


def combine_preds(predictions, weight_matrix):
    """Combine predictions using given weights"""

    predictions = np.average(
        predictions, axis=3, weights=weight_matrix)
    combined_predictions = []
    combined_confidences = []
    for pred in tqdm(predictions, desc="Converting predictions to events", unit="window"):
        # Convert to relative window event timestamps
        events = pred_to_event_state(pred, thresh=0, n_events=10)

        # Add step offset based on repeat factor.
        if data_info.downsampling_factor <= 1:
            offset = 0
        elif data_info.downsampling_factor % 2 == 0:
            offset = (data_info.downsampling_factor / 2.0) - 0.5
        else:
            offset = data_info.downsampling_factor // 2
        steps = (events[0] + offset, events[1] + offset)
        confidences = (events[2], events[3])
        combined_predictions.append(steps)
        combined_confidences.append(confidences)

    # Return tuple
    return combined_predictions, combined_confidences


def get_objective(config_loader: ConfigLoader):
    """Returns a trial function that takes an optuna trial to choose weights and returns a score.
    Wrapped so it does not recompute the predictions."""

    # Get ensemble
    ensemble = config_loader.get_ensemble()

    store_location = config_loader.get_model_store_loc()

    # Make predictions
    predictions_all = pred_all(ensemble, store_location)

    def objective(trial: optuna.trial):
        # Suggest weights for ensemble
        num_weights = len(config_loader.config["ensemble"]["weights"])
        weight_matrix = [
            trial.suggest_float("weight_" + str(i), 0, 1)
            for i in range(num_weights)
        ]

        logger.info("Weighting predictions with confidences")
        predictions = combine_preds(predictions_all, weight_matrix)

        logger.info("Formatting predictions...")
        test_ids = ensemble.get_test_ids()
        test_window_info = data_info.window_info[data_info.window_info['series_id'].isin(test_ids)]

        submission = to_submission_format(predictions, test_window_info)

        # load solution for test set and compute score
        solution = (pd.read_csv(config_loader.get_train_events_path())
                    .groupby('series_id')
                    .filter(lambda x: x['series_id'].iloc[0] in test_ids)
                    .reset_index(drop=True))
        logger.info("Start scoring test predictions...")
        return compute_score_full(submission, solution)

    return objective


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install()

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Load config file
    config_loader: ConfigLoader = ConfigLoader("config.json")
    config_hash = hash_config(config_loader.get_config(), length=16)

    study = optuna.create_study(
        study_name="detect-sleep-states" + config_hash,
        storage="sqlite:///optuna.db",
        load_if_exists=False,
        direction="maximize",
    )
    study.optimize(get_objective(config_loader), n_trials=100)
    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)
