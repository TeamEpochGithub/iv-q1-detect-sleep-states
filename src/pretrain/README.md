# Pretrain

This step includes preparing the data for inputting in the model. It contains downsampling options, standardization and
train-test split.

## Pretraining

This step includes preparing the data for inputting in the model.
It contains downsampling options, standardization and train-test split.

List of options:

- `downsample`: Downsamples all features
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