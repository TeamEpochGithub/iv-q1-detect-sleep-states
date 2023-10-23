# Pretrain

This step includes preparing the data for inputting in the model. It contains downsampling options, standardization and train-test split.

## Pretraining
This step prepares the data for inputting it in the model.
It can split the data into train and test sets, and standardize the data.

List of options:

- `downsample`: Downsamples all features
    - `factor`: downsampling factor
    - `features`: Any existing numerical features
    - `methods`: ["min", "max", "mean", "std", "median"]
    - `standard`: "mean" | "median"
- `remove_features`
    - `features`: Any existing numerical features
- `test_size` ∈ [0, 1]: percentage of data to be used for testing.
- `scaler`: method used for standardization. See [Scalers](../scaler/README.md) for more info.

Example:
```JSON
"pretraining": {
    "downsample": {
        "factor": 12,
        "features": ["anglez", "enmo"],
        "methods": ["min", "max", "mean", "std", "median"],
        "standard": "mean"
    },
    "remove_features": ["anglez", "enmo"],
    "test_size": 0.2,
    "scaler": {
        "kind": "standard-scaler",
        "copy": true
    }
}
```