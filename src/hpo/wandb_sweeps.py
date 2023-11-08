from collections.abc import Callable

import wandb

from src.hpo.hpo import HPO
from src.logger.logger import logger
from src.util.hash_config import hash_config


class WandBSweeps(HPO):
    """Hyperparameter optimization using Weights & Biases Sweeps."""

    def __init__(self, sweep_configuration: dict[str], count: int | None = None) -> None:
        """Initialize the hyperparameter optimization method.

        :param sweep_configuration: the configuration for the Weights & Biases Sweeps
        :param count: the number of runs to execute, or None to run infinitely
        """
        super().__init__()
        self.sweep_configuration = sweep_configuration
        self.count = count

        model_name: str = list(sweep_configuration["parameters"]["models"]["parameters"].keys())[0]
        self.sweep_configuration["name"] = f"{model_name}/{hash_config(sweep_configuration, length=16)}"
        self.sweep_id = wandb.sweep(sweep=self.sweep_configuration, project="detect-sleep-states")

    def optimize(self, to_optimize: Callable) -> None:
        """Optimize the hyperparameters for a single model with Weights & Biases Sweeps.

        Gotta Sweep, Sweep, Sweep!

        :param to_optimize: the function that runs the preprocessing, feature engineering, pretrain, training, and cross validation.
        """
        logger.info("Optimizing hyperparameters with Weights & Biases Sweeps")
        wandb.agent(self.sweep_id, function=to_optimize, count=self.count)
        logger.info("Hyperparameter optimization complete")
