import optuna
import pandas as pd
import torch

from src import data_info
from src.configs.load_config import ConfigLoader
from src.logger.logger import logger
from src.score.compute_score import compute_score_full
from src.util.hash_config import hash_config
from src.util.submissionformat import to_submission_format


def get_objective(config_loader: ConfigLoader):
    def objective(trial: optuna.trial):
        pred_cpu = False

        # Get ensemble
        ensemble = config_loader.get_ensemble()

        # Suggest weights for ensemble
        num_weights = len(config_loader.config["ensemble"]["weights"])
        config_loader.config["ensemble"]["weights"] = [
            trial.suggest_float("weight_" + str(i), 0, 1)
            for i in range(num_weights)
        ]

        # Make predictions on test data
        predictions = ensemble.pred(
            config_loader.get_model_store_loc(), pred_with_cpu=pred_cpu)
        test_ids = ensemble.get_test_ids()

        logger.info("Formatting predictions...")
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
