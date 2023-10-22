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
7. [Ensemble](#ensemble)
8. [Loss](#loss)
9. [Hyperparameter optimization](#hyperparameter-optimization)
10. [Cross validation](#cross-validation)
11. [Scoring](#scoring)

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
    - Parameters: `id_encoding_path: str`, `events_path: str`
    - Labels the data in a way that each timestep gets a label.
        - `0`: asleep.
        - `1`: awake.
        - `2`: `NaN`, not labeled.
- `split_windows`
    - Parameters: `start_hour: int = 15`, `window_size: int = 17280`
    - Splits the data in to 24 hour long windows
- `remove_unlabeled` (requires `add_state_labels`, optional `split_windows`)
    - Removes all the data points where there is no labeled data
- `truncate` (requires `add_state_labels`)
    - Truncates the unlabeled end of the data
    - `remove_unlabeled` also removes the unlabeled end
- `add_regression_labels` (requires `add_state_labels`, `split_windows`)
    - Adds, the wakeup, onset, wakeup-NaN and onset-NaN labels
- `add_segmentation_labels` (requires `add_state_labels`)
    - Adds 3 columns, hot encoded, for the segmentation labels
        - 0: `hot-asleep`
        - 1: `hot-awake`
        - 2: `hot-NaN`, not labeled

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
        "events_path": "data/raw/train_events.csv"
    },
    {
        "kind": "split_windows",
        "start_hour": 15,
        "window_size": 17280
    },
    {
        "kind": "remove_unlabeled"
    },
    {
        "kind": "truncate"
    },
    {
        "kind": "add_regression_labels"
    },
    {
        "kind": "add_segmentation_labels"
    }
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
Features that should be included during training and submission. 

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
    - `day`: true | false (opt)
    - `hour`: true | false (opt)
    - `minute`: true | false (opt)
    - `second`: true | false (opt)
- `rotation`
    - `window_sizes`: a list of sizes for rolling median smoothing, classic baseline uses 100

Example:
```JSON
"feature_engineering": {
    "fe1": {
        "window_sizes": [5, 10],
        "features": ["enmo", "anglez"]
    },
    "time": {
        "day": true,
        "hour": true,
        "minute": false,
        "second": false
    }
}
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
This step prepares the data for inputting it in the model.
It can split the data into train and test sets, and standardize the data.

List of options:
- `test_size` ∈ [0, 1]: percentage of data to be used for testing.
- `scaler`: method used for standardization. See [Scalers](../scaler/README.md) for more info.

Example:
```JSON
"pretraining": {
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

- regression-transformer
    - epochs (required)
    - loss (required)
    - optimizer (required)
    - lr=0.001
    - batch_size=32
    - patch_size=36
    - feat_dim=patch_size*num_features
    - max_len=window_size
    - d_model=x (x * n_heads)
    - n_heads=6
    - num_layers=5
    - dim_feedforward=2048
    - num_classes=4 (Points to regress to)
    - dropout=0.1
    - pos_encoding='learnable' ["learnable", "fixed"]
    - activation="relu" ["relu", "gelu"]
    - norm="BatchNorm" ["BatchNorm", "LayerNorm"]
    - freeze=False

- classic-base-model
  - median_window=100
  - threshold=.1

- event-nan-regression-transformer
    - epochs_events (required)
    - epochs_nans (required)
    - loss_events (required)
    - loss_nans (required)
    - optimizer_events (required)
    - optimizer_nans (required)
    - lr_events=0.000035
    - lr_nans=0.000035
    - batch_size=16
    - patch_size=36
    - feat_dim=patch_size*num_features
    - max_len=window_size
    - d_model=x (x * n_heads)
    - n_heads=6
    - num_layers=5
    - dim_feedforward=2048
    - num_classes=2 (Points to regress to)
    - dropout=0.1
    - pos_encoding='learnable' ["learnable", "fixed"]
    - act_int="relu" ["relu", "gelu"]
    - act_out="relu" ["relu", "gelu", "sigmoid"]
    - norm="BatchNorm" ["BatchNorm", "LayerNorm"]
    - freeze=False

- event-regression-transformer
    - epochs (required)
    - loss (required)
    - optimizer (required)
    - lr=0.000035
    - batch_size=16
    - patch_size=36
    - max_len=window_size
    - d_model=x (x * n_heads)
    - n_heads=6
    - num_layers=5
    - dim_feedforward=2048
    - num_classes=2 (Points to regress to)
    - dropout=0.1
    - pos_encoding='learnable' ["learnable", "fixed"]
    - act_int="relu" ["relu", "gelu"]
    - act_out="relu" ["relu", "gelu", "sigmoid"]
    - norm="BatchNorm" ["BatchNorm", "LayerNorm"]
    - freeze=False

  
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
"EventNanRegressionTransformer": {
    "type": "event-nan-regression-transformer",
    "epochs_events": 20,
    "epochs_nans": 20,
    "loss_events": "event-regression-mae",
    "loss_nans": "nan-regression",
    "optimizer_events": "adam-torch",
    "optimizer_nans": "adam-torch",
    "lr_events": 0.000035,
    "lr_nans": 0.000035,
    "batch_size": 16,
    "patch_size": 36,
    "feat_dim": 72,
    "max_len": 480,
    "d_model": 480,
    "n_heads": 6,
    "num_layers": 5,
    "dim_feedforward": 256,
    "num_classes": 2,
    "dropout": 0.1,
    "pos_encoding": "fixed",
    "act_int": "relu",
    "act_out": "relu",
    "norm": "BatchNorm",
    "freeze": false
}
"EventTransformer": {
    "type": "event-regression-transformer",
    "epochs": 100,
    "loss": "event-regression-rmse",
    "optimizer": "adam-torch",
    "lr": 0.000035,
    "batch_size": 16,
    "patch_size": 30,
    "max_len": 576,
    "d_model": 96,
    "n_heads": 6,
    "num_layers": 5,
    "dim_feedforward": 2048,
    "num_classes": 2,
    "dropout": 0.1,
    "pos_encoding": "fixed",
    "act_int": "relu",
    "act_out": "relu",
    "norm": "BatchNorm",
    "freeze": false
}
"Classic-baseline": {
    "type": "classic-base-model",
    "median_window": 100,
    "threshold": .1
}

"1D-Unet-CNN": {
    "type": "seg-unet-1d-cnn",
    "loss": "ce-torch",
    "optimizer": "adam-torch",
    "epochs": 15,
    "batch_size": 32,
    "lr": 0.001,
    "hidden_layers": 32,
    "early_stopping": 5
}
```

## Model store location

Specify location where models should be stored, furthermore, the config should be stored together

```JSON
"model_store_loc": "./tm",
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
Will be implemented in the future. The plan is to automatically detect if multiple model values are given, and then
applying a hyperparameter optimization.

Specification for what to do for hyperparameter optimization

```JSON
"hpo": {
    "apply": true | false,
    "method": "hpo1"
}
```

Options

- hpo1
- hpo2

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