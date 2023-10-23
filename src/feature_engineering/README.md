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

    - Example:
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
            "day": true,
            "hour": true,
            "minute": false,
            "second": false        
        }
],
```