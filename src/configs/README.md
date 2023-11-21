# Config
The config file is a JSON file that contains all the information needed to run the pipeline. 
The config file is split up in different sections. Each section has its own config options.

## Config Options
1. [Preprocessing steps](#preprocessing-steps)
2. [Preprocessing data location](#preprocessing-data-location)
3. [Feature engineering](#feature-engineering)
4. [Feature engineering data location](#feature-engineering-data-location)
5. [Pretraining](#pretraining)
6. [Models](#models)
7. [Model store location](#model-store-location)
8. [Cross validation](#cross-validation)
9. [Ensemble](#ensemble)
10. [Loss](#loss)
11. [Hyperparameter optimization](#hyperparameter-optimization)
12. [Scoring](#scoring)
13. [Train for submission](#train-for-submission)
14. [Visualize preds](#visualize-preds)
15. [Similarity filtering](#similarity-filtering)

## General
- `name`: `str`
    - Name of the config
- `log_to_wandb`: `bool`
    - Whether to log to wandb or not
- `pred_with_cpu`: `bool`
    - Whether to predict with cpu or not
- `train_series_path`: `str`
    - Path to the series to train on
- `train_events_path`: `str`
    - Path to the events to train on
- `test_series_path`: `str`
    - Path to the series to test on

## Data Info
This is where global variables are stored that we want to use across multiple files.

Variables:
- `window_size`: `int`
    - The size of the window in steps before donwsampling. Default is 24 * 60 * 12 = 17280
- `downsampling_factor`: `int`
  - The factor to downsample the data with. Default is 1, which means no downsampling.

## Preprocessing steps
These steps are executed in order placed in the list. 
The order is important as some steps depend on the output of previous steps.
For more information on the preprocessing steps, see [Preprocessing](../preprocessing/README.md).

The following steps are currently implemented:

- `mem_reduce`
    - Parameters: `id_encoding_path: Optional[str] = None`
    - Reduces the memory usage of the dataframe. Encodes the series IDs to unique ints and converts the timestamp to
      a datetime object.
    - Stores the series ID encoding in the specified path. If there is no path specified, it will not store the encoding.
- `similarity_nan`
    - Compute similarity between all windows to detect a repeating pattern indicating NaN. 
Adds this as a column that is 0 for perfect similarity.
- `add-noise`
    - Adds gaussian noise to the sensor data.
- `add_state_labels`
    - Parameters: `id_encoding_path: str`, `events_path: str`, `use_similarity_nan: bool`, `fill_limit: int`
    - Labels the data in a way that each timestep gets a label. (Column name: awake)
        - `0`: asleep.
        - `1`: awake.
        - `2`: `NaN`, not labeled.
    - If `use_similarity_nan` is set to true, 
      it will use the similarity_nan column to determine whether something is NaN or unlabeled, 
      and asleep/awake labelling around those segments will be better.
    - `fill_limit` is the maximum number of time steps that a state is extended into unlabeled non-nan data,
      only used if `use_similarity_nan` is set to true.
    - `nan_tolerance_window`, labels are extended up to the first nan data point. 
      This parameter specifies the size of the median filter that is used to ignore lone nan points.
- `split_windows`
    - Parameters: `start_hour: int = 15`
    - Splits the data in to 24 hour long windows
- `remove_unlabeled` (requires `add_state_labels`, optional `split_windows`)
    - Parameters: `remove_entire_series: bool`, `remove_partially_unlabeled_windows: bool`, `remove_nan`
    - Removes all the unlabeled data points and optionally, all NaN data. 
- `add_regression_labels` (requires `split_windows`)
    - Parameters: `id_encoding_path: str`, `events_path: str`,
    - Adds 3 columns, the wakeup, onset, wakeup-NaN and onset-NaN labels
        - 0: `wakeup`
        - 1: `onset`
        - 2: `wakeup-NaN`
        - 3: `onset-NaN`
- `add_segmentation_labels` (requires `add_state_labels`)
    - Adds 3 columns, hot encoded, for the segmentation labels
        - 0: `hot-asleep`
        - 1: `hot-awake`
        - 2: `hot-NaN`
        - 3: `hot-unlabeled`
- `add_event_labels`
    - Parameters: `id_encoding_path: str`, `smoothing: int`, `events_path: str`
    - Adds 2 columns, 0 for no event and 1 for event and applies gaussian smoothing over it
        - 0: `state_onset`
        - 1: `state_wakeup`

Example for each step:
```JSON
"preprocessing": [
    {
        "kind": "mem_reduce",
        "id_encoding_path": "series_id_encoding.json"
    },
    {
        "kind": "add_noise"
    },
    {
        "kind": "add_state_labels",
        "id_encoding_path": "series_id_encoding.json",
        "events_path": "data/raw/train_events.csv",
        "use_similarity_nan": true,
        "fill_limit": 8640,
        "nan_tolerance_window": 5
      },
    {
        "kind": "split_windows",
        "start_hour": 15
    },
    {
        "kind": "remove_unlabeled",
        "remove_entire_series": false,
        "remove_partially_unlabeled_windows": true,
        "remove_nan": true
    },
    {
        "kind": "truncate"
    },
    {
        "kind": "add_regression_labels",
        "id_encoding_path": "series_id_encoding.json",
        "events_path": "data/raw/train_events.csv"
    },
    {
        "kind": "add_segmentation_labels"
    },
    {
        "kind": "add_event_labels",
        "id_encoding_path": "series_id_encoding.json",
        "events_path": "data/raw/train_events.csv",
        "smoothing": 5
    },
]
```

## Preprocessing data location
Location out: Data created by preprocessing is stored in this directory. 

Location in: Data needed by preprocessing is stored in this directory. 

```JSON
"processed_loc_out": "./data/processed",
"processed_loc_in": "./data/raw"
```

## Feature engineering
Features that should be included during training and submission. It contains a list of feature engineering steps, applied in order, where each step is a dictionary.

List of options and their config options: 

- `kurtosis`
    - `window_sizes` > 3
    - `features`: Any existing numerical features
- `mean`
    - `window_sizes` > 3
    - `features`: Any existing numerical features
- `skewness`
    - `window_sizes` > 3
    - `features`: Any existing numerical features
- `time`
    - `time_features`: a list of time features to include
      - Options: `year`, `month`, `day`, `hour`, `minute`, `second`, `microsecond`
- `rotation`
    - `window_sizes`: a list of sizes for rolling median smoothing, classic baseline uses 100
- `similarity_nan`
  - `as_feature`: Boolean that if True, names the column "f_similarity_nan", else just "similarity_nan"
- `sun`
    -  `sun_features`: a list of sun features to include based the longitude and latitude that should be specified in the data_info.
    Currently, NYC location is used.
    - Options: `elevation`, `azimuth`
- `add_holidays`

Example:
```JSON
"feature_engineering": [
        {
            "kind": "rotation",
            "window_sizes": [100]
        },
        {
            "kind": "kurtosis",
            "window_sizes": [5, 10],
            "features": ["anglez", "enmo"]
        },
        {
            "kind": "mean",
            "window_sizes": [5, 10],
            "features": ["anglez", "enmo"]
        },
        {
            "kind": "skewness",
            "window_sizes": [5, 10],
            "features": ["anglez", "enmo"]
        },
        {
            "kind": "time",
            "time_features": ["day", "hour", "minute", "second"]    
        }
        {
            "kind": "similarity_nan",
            "as_feature": true,
        },
        {
            "kind": "sun",
            "sun_features": ["elevation", "azimuth"]
        },
        {
            "kind": "add_holidays"
        }
]
```

## Feature engineering data location

<p>
Location out: Data created by feature engineering is stored in this location <br>
Location in: Data needed by feature engineering is stored in this location
</p>

```JSON
"fe_loc_out": "./data/features"
"fe_loc_in": "./data/processed"
```

## Pretraining
This step includes preparing the data for inputting in the model. 
It contains downsampling options, standardization and train-test split.

List of options:

- `downsample`: Downsamples all features
    - `factor`: downsampling factor
    - `features`: Any existing numerical features
    - `methods`: ["min", "max", "mean", "std", "median", "var", "sum"]
    - `standard`: "min" | "max" | "mean" | "std" | "median" | "var" | "sum"
- `test_size` ∈ [0, 1]: percentage of data to be used for testing.
- `scaler`: method used for standardization. See [Scalers](../scaler/README.md) for more info.

Example:
```JSON
"pretraining": {
    "downsample": {
        "features": ["anglez", "enmo"],
        "methods": ["min", "max", "mean", "std", "median"],
        "standard": "mean"
    },
    "test_size": 0.2,
    "scaler": {
        "kind": "standard-scaler",
        "copy": true
    }
}
```

## Models
A list of models and their specified configurations are included here. Multiple can be entered as this allows for the
creation of ensembles. Additionally, the location they should be stored is included.
These models should either be a statistical, regression or state_prediction model that predicts the current timestep

```JSON
"models": {
    "model1name": {
        MODEL SPECIFIC CONFIG OPTIONS
    },
    "model2name": {
        MODEL SPECIFIC CONFIG OPTIONS
    }
}
```


#### Implemented Models types and config options

This contains all the models and their hyperparameters that are implemented. The config options are the hyperparameters that are standard.


##### Basic models
- example-fc-model
    - epochs (required)
    - loss (required)
    - optimizer (required)
    - lr
    - batch_size


- seg-simple-1d-cnn
    - loss (required)
    - optimizer (required)
    - epochs
    - lr
    - batch_size


- classic-base-model
  - median_window=100
  - threshold=.1
  - use_nan_similarity=True

##### State segmentation models
- seg-unet-1d-cnn (one-hot state segmentation model that predicts awake, asleep, NaN for each timestep)
    - loss (required, ce-torch recommended)
    - optimizer (required)
    - epochs=10
    - batch_size=32
    - lr=0.001
    - hidden_layers=32
    - kernel_size=7 (only works on 7 for now)
    - depth=2
    - early_stopping=-1
    - weight_decay=0.0
    - mask_unlabeled=False

##### Event segmentation models
- event-seg-unet-1d-cnn / split-event-seg-unet-1d-cnn (event segmentation model that predicts state-onset, state-wakeup) for each timestep
    - loss (required, shrinkage-loss recommended)
    - optimizer (required)
    - epochs=10
    - batch_size=32
    - lr=0.001
    - network_params (hidden_layers=32, depth=2, kernel_size=7, activation=relu)
    - activation_delay (number of epochs to wait before applying activation to last layer)
    - lr_schedule (config for learning rate schedule, see [CosineLRWithRestarts](https://timm.fast.ai/SGDR))
    - early_stopping=-1
    - weight_decay=0.0
    - threshold=0 (threshold for the event prediction, if the prediction is below this, it returns a nan)
    - mask_unlabeled=false

- event-res-gru
    - epochs (required) 
    - loss (required)
    - optimizer (required)
    - early_stopping
    - network_params (hidden_size, n_layers, activation)
    - activation_delay (number of epochs to wait before applying activation to last layer)
    - lr
    - lr_schedule (config for learning rate schedule, see [CosineLRWithRestarts](https://timm.fast.ai/SGDR))
    - threshold

#### Transformers
- transformer / segmentation-transformer / event-segmentation-transformer
    - epochs (required) 
    - loss (required)
    - optimizer (required)
    - early_stopping
    - network_params (heads, emb_dim, forward_dim": 96,
        "n_layers": 6,
        "pooling": "none",
        "tokenizer": "patch",
        "tokenizer_args": {
            "patch_size": 12
        },
        "pe": "other",
        "dropout": 0.0)
    - activation_delay (number of epochs to wait before applying activation to last layer)
    - lr
    - lr_schedule (config for learning rate schedule, see [CosineLRWithRestarts](https://timm.fast.ai/SGDR))
    - threshold
  
Example of an example-fc-model configuration and a 1D-CNN configuration

```JSON
"ExampleModel": {
    "type": "example-fc-model",
    "epochs": 20,
    "batch_size": 32,
    "loss": "mae-torch",
    "optimizer": "adam-torch"
}
"1D-CNN": {
    "type": "seg-simple-1d-cnn",
    "loss": "mse-torch",
    "optimizer": "rmsprop-torch",
    "epochs": 5,
    "batch_size": 64,
    "lr": 0.01
}

"GeneralTransformer": {
    "type": "transformer" / "segmentation-transformer" / "event-segmentation-transformer",
    "epochs": 5,
    "loss": "event-regression-rmse",
    "optimizer": "adam-torch",
    "lr": 0.00035,
    "batch_size": 16,
    "network_params": {
        "heads": 8,
        "emb_dim": 48,
        "forward_dim": 96,
        "n_layers": 6,
        "pooling": "none",
        "tokenizer": "patch",
        "tokenizer_args": {
            "patch_size": 12
        },
        "pe": "other",
        "dropout": 0.0
    }
}

"Classic-baseline": {
    "type": "classic-base-model",
    "median_window": 100,
    "threshold": 0.1,
    "use_nan_similarity": true
}

"1D-Unet-CNN": {
    "type": "seg-unet-1d-cnn",
    "loss": "bce-torch",
    "optimizer": "adam-torch",
    "epochs": 2,
    "batch_size": 32,
    "lr": 0.001,
    "hidden_layers": 8
}


"EventResGRU": {
    "type": "event_res_gru",
    "loss": "shrinkage-loss",
    "epochs": 100,
    "batch_size": 32,
    "optimizer": "adam-torch",
    "early_stopping": 10,
    "network_params": {
      "hidden_size": 64,
      "n_layers": 5,
      "activation": "relu"
    },
    "activation_delay": 15,
    "lr": 0.001,
    "lr_schedule": {
      "t_initial": 100,
      "warmup_t": 5,
      "warmup_lr_init": 1e-6,
      "lr_min": 2e-8
    },
    "threshold": 0

"Event-1D-Unet-CNN": {
      "type": "split-event-seg-unet-1d-cnn" / "event-seg-unet-1d-cnn",
      "loss": "shrinkage-loss",
      "optimizer": "adam-torch",
      "epochs": 5,
      "early_stopping": 10,
      "batch_size": 32,
      "lr": 0.001,
      "network_params": {
          "hidden_layers": 8,
          "activation": "relu"
      },
      "lr_schedule": {
          "t_initial": 100,
          "warmup_t": 5,
          "warmup_lr_init": 0.000001,
          "lr_min": 2e-5
      },
      "activation_delay": 5,
      "threshold": 0
}
```

## Model store location

Specify location where models should be stored, furthermore, the config should be stored together

```JSON
"model_store_loc": "./tm",
```

## Cross validation
### Splitters
A splitter splits the data into train and test sets. 
The splitters for groups are likely most relevant for our data.

List of splitter options with parameters (see [sklearn.model_selection](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection) for more details): 
- `group_k_fold`
  - `n_splits: int`, default = 5
- `group_shuffle_split`
  - `n_splits: int`, default = 5
  - `test_size: float`, default = 0.2
  - `train_size: float | int`, default = None
  - `random_state: int`, default = None
- `k_fold`
  - `n_splits: int`, default = 5
  - `shuffle: bool`, default = False
  - `random_state: int`, default = None
- `leave_one_group_out`
- `leave_p_group_out`
  - `n_groups: int`, default = 2
- `leave_one_out`
- `leave_p_out`
  - `p: int`
- `predefined_split`
  - `test_fold: array-like` of shape `(n_samples,)`
- `shuffle_split`
  - `n_splits: int`, default = 5
  - `test_size: float`, default = 0.2
  - `train_size: float | int`, default = None
  - `random_state: int`, default = None
- `stratified_k_fold`
  - `n_splits: int`, default = 5
  - `shuffle: bool`, default = False
  - `random_state: int`, default = None
- `stratified_shuffle_split`
  - `n_splits: int`, default = 10
  - `test_size: float | int`, default = None
  - `train_size: float | int`, default = None
  - `random_state: int`, default = None
- `stratified_group_k_fold`
  - `n_splits: int`, default = 5
  - `shuffle: bool`, default = False
  - `random_state: int`, default = None
- `time_series_split`
  - `n_splits: int`, default = 5
  - `max_train_size: int`, default = None
  - `test_size: int`, default = None
  - `gap: int`, default = 0

### Scorers
List of scorer options:
- `score_full`: computes the score for the full dataset
- `score_clean` computes the score for the clean dataset

Currently, the scorers have only been tested with the `seg-unet-1d-cnn`.

### Usage
Example:
```JSON
"cv": {
    "splitter": "group_shuffle_split",
    "scoring": ["score_full", "score_clean"],
    "splitter_params": {
        "n_splits": 1,
        "test_size": 0.2
    }
}
```

## Ensemble
For now, we support just an ensemble of 1 function.

Ensemble specifications including the models used, the weight of each, and how the model predictions should be combined

```JSON
"ensemble": {
    "models": ["model1name", "model2name"],
    "weights": [1, 2],
    "comb_method": "addition",
}
```

Combination methods

- Addition
- Max confidence

## Loss
Current loss functions that are implemented. Returns a LossException if a loss function has not been found.
Example:

```JSON
"loss": "mse-torch"
```

Options

- "mse-torch"
- "mae-torch"
- "crossentropy-torch"
- "binarycrossentropy-torch"

## Hyperparameter optimization
See [HPO Readme](../hpo/README.md) for more information.

### Usage
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



## Scoring
Choose whether to do the scoring and show plots

```JSON
"scoring": True | False
```

## Train for submission
Once we have a model that we want to use for submission, we can train it on all the data we have available. This is done
by setting the following to true:

```JSON
"train_for_submission": True
```

## Visualize preds
Configures how plots are generated. 
- "n": Int that specifies the number of plots to generate (for saving jpegs and plotly plots)
- "browser_plot": Boolean that if set to True creates plotly plots
- "save": Boolean that if set to True saves the images  

```
"visualize_preds": {
        "n": 5,
        "browser_plot": true,
        "save": false
    }
```

## Similarity filtering
Apply a filter to the timestamp predictions, based on similarity to other windows.
Requires the similarity_nan preprocessing step. Can be removed from the 
If the proportion of steps that are perfectly similar to another window is above the threshold, prediction is set to nan.
- "threshold": Float, if mean of f_similarity_nan is above this, prediction is set to nan

```JSON
"similarity_filter": {
        "threshold": 0.27
    }
```
