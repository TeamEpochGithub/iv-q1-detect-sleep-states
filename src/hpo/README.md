# Hyperparameter Optimization (HPO)
This package contains code related to HPO.

Currently, the following HPO methods are implemented:
- Weight & Biases Sweeps
  - Grid Search
  - Random Search
  - Bayesian Optimization

## Usage
If the function that runs the experiment is called `main`, then the following code can be used to run the experiment with or without HPO:
```python
config_loader: ConfigLoader = ConfigLoader("config.json")
hpo: HPO | None = config_loader.hpo

if hpo is None:  # HPO disabled
    main()
else:  # HPO enabled
    hpo.optimize(main)
```

Whether `config_loader.hpo` returns `None` or an instance of `HPO` depends on the configuration.
It only returns `None` if HPO is disabled. 

## Configuration
- `"kind": str`: The kind of HPO to use. Currently, only `"wandb_sweeps"` is supported.
- `"apply": bool`: Whether to apply the HPO. If `false`, the HPO is disabled and `ConfigLoader.hpo` returns `None`.
- `"count": int`: The number of runs to perform. If `count` is missing, it will run indefinitely. 
- `"sweep_configuration": dict`: The configuration for the HPO method if it is `"wandb_sweeps"`.
  See the [Weights & Biases documentation](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration) for more information.

Here's an example configuration for the `SplitEvent-1D-Unet-CNN` model 
where we want to optimize the number of epochs using random search with Weights & Biases Sweeps:
```json
"hpo": {
    "kind": "wandb_sweeps",
    "apply": true,
    "count": 2,
    "sweep_configuration": {
        "method": "random",
        "metric": {"goal": "maximize", "name": "cv_score"},
        "parameters": {
            "models": {
                "parameters": {
                    "SplitEvent-1D-Unet-CNN": {
                        "parameters": {
                            "type": {"value": "split-event-seg-unet-1d-cnn"},
                            "loss": {"value": "mse-torch"},
                            "optimizer": {"value": "adam-torch"},
                            "epochs": {"values": [3, 5, 7]},
                            "batch_size": {"value": 32},
                            "lr": {"value": 0.0005},
                            "hidden_layers": {"value": 8},
                            "early_stopping": {"value": 7},
                            "threshold": {"value": 0}
                        }
                    }
                }
            }
        }
    }
},
```
