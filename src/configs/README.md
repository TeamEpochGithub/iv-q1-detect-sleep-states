## Config options

1. [Preprocessing steps](#preprocessing-steps)
2. [Preprocessing data location](#preprocessing-data-location)
3. [Feature engineering](#feature-engineering)
4. [Feature engineering data location](#feature-engineering-data-location)
5. [Models](#models)
6. [Ensemble](#ensemble)
7. [Loss](#loss)
8. [Hyperparameter optimization](#hyperparameter-optimization)
9. [Cross validation](#cross-validation)
10. [Scoring](#scoring)

### Preprocessing steps

These steps are executed in the order placed in the dictionary

List of options and what they do
- "add-noise"
    - Adds gaussian noise to the sensor data
- "add_state_labels"
    - Labels the data in a way that each timestep gets a label. 0: asleep, 1: awake, 2: NaN, not labeled
- "mem_reduce"
    - Reduces the memory usage of the dataframe. Encodes the series IDs to unique ints and converts the timestamp to
    a datetime object
- "split_windows"
    - Splits the data in to 24 hour long windows

Example:
```
"preprocessing": ["pp1", "pp2"]
```

List of options:

- `mem_reduce`
- `add_noise`
- `add_state_labels`
- `split_windows` (currently 24h hardcoded)
- `remove_unlabeled`
- `truncate`

### Preprocessing data location

<p>
Location out: Data created by preproccesing is stored in this location <br>
Location in: Data needed by preprocessing is stored in this location
</p>

```
"processed_loc_out": "./data/processed"
"processed_loc_in": "./data/raw"
```

List of options and what they do

- `mem_reduce`
    - Reduces the memory usage of the dataframe. Encodes the series IDs to unique ints and converts the timestamp to
      a datetime object
- `add-noise`
    - Adds gaussian noise to the sensor data
- `add_state_labels`
    - Labels the data in a way that each timestep gets a label. 0: asleep, 1: awake, 2: `NaN`, not labeled
- `split_windows`
    - Splits the data in to 24 hour long windows
- `remove_unlabeled`
    - Removes all the data points where there is no labeled data
- `add_events_labels`
    - Adds, the wakeup, onset, wakeup-NaN and onset-NaN labels
- `truncate`
    - Truncates the unlabeled end of the data
    - `remove_unlabeled` also removes the unlabeled end

### Feature engineering

Features that should be included during training and submission

List of options and their config options

- "kurtosis"
    - "window_sizes": x > 3
    - "features": Any existing numerical features
- "mean"
    - "window_sizes": x > 3
    - "features": Any existing numerical features
- "skewness"
    - "window_sizes": x > 3
    - "features": Any existing numerical features

Example:
``` 
"feature_engineering": {
    "fe1": {
        "window_sizes": [5, 10],
        "features": ["enmo", "anglez"]
    },
    "fe2": {}
    }
```

### Feature engineering data location

<p>
Location out: Data created by feature engineering is stored in this location <br>
Location in: Data needed by feature engineering is stored in this location
</p>

``` 
"fe_loc_out": "./data/features"
"fe_loc_in": "./data/processed"
```

### Pre-training step

This step includes preparing the data for inputting in the model.
List of options and their config options

- "cv": > 0 (number of folds)
- "test_size": > 0 (percentage of data to be used for testing)
- "standardize": method used for standardization
    - "minmax"
    - "standard"

You are not able to select cv and train_test_split at the same time.

Example:

```
"pre_training": {
    "cv": 5, || "test_size": 0.2,
    "standardize": "standard"
}
```

### Models

A list of models and their specified configurations are included here. Multiple can be entered as this allows for the
creation of ensembles. Additionally, the location they should be stored is included.
These models should either be a statistical, regression or state_prediction model that predicts the current timestep

``` 
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

It should

- example-fc-model
    - epochs (required)
    - loss (required)
    - optimizer (required)
    - lr
    - batch_size

Example of an example-fc-model configuration:

```
"ExampleModel": {
    "type": "example-fc-model",
    "epochs": 20,
    "batch_size": 32,
    "loss": "mae-torch",
    "optimizer": "adam-torch"
}
```

### Model store location

Specify location where models should be stored, furthermore, the config should be stored together

```
"model_store_loc": "./tm",
```

### Ensemble

For now, we support just an ensemble of 1 function.

Ensemble specifications including the models used, the weight of each, and how the model predictions should be combined

```
"ensemble": {
    "models": ["model1name", "model2name"],
    "weights": [1, 2],
    "comb_method": "addition",
}
```

Combination methods

- Addition
- Max confidence

### Loss

Current loss functions that are implemented. Returns a LossException if a loss function has not been found.
Example:

```
    "loss": "mse-torch"
```

Options

- "mse-torch"
- "mae-torch"
- "crossentropy-torch"
- "binarycrossentropy-torch"

### Hyperparameter optimization

Will be implemented in the future. The plan is to automatically detect if multiple model values are given, and then
applying a hyperparameter optimization.

Specification for what to do for hyperparameter optimization

```
"hpo": {
    "apply": true | false,
    "method": "hpo1"
}
```

Options

- hpo1
- hpo2

### Scoring

Choose whether to do the scoring and show plots

```
"scoring": True | False
```

### Train for submission

Once we have a model that we want to use for submission, we can train it on all the data we have available. This is done
by setting the following to true:

```
"train_for_submission": True
```