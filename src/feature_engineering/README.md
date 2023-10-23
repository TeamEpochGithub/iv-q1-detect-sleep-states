# This directory contains scripts related to feature engineering

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
    - `day`: true | false (opt)
    - `hour`: true | false (opt)
    - `minute`: true | false (opt)
    - `second`: true | false (opt)
- `rotation`
    - `window_sizes`: a list of sizes for rolling median smoothing, classic baseline uses 100
- `downsample`
    - `factor`: downsampling factor
    - `features`: Any existing numerical features
    - `methods`: ["min", "max", "mean", "std", "median"]
    - `standard`: "mean" | "median"
- `remove_enmo`
- `remove_anglez`


Example:
```JSON
"feature_engineering": [
        {
            "kind": "downsample",
            "factor": 12,
            "features": ["anglez", "enmo"],
            "methods": ["min", "max", "mean", "std", "median"],
            "standard": "mean"
        },
        {
            "kind": "remove_enmo"
        },
        {
            "kind": "remove_anglez"
        },
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
            "day": true,
            "hour": true,
            "minute": false,
            "second": false        
        }
],
```