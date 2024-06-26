## Feature engineering

Features that should be included during training and submission. It contains a list of feature engineering steps,
applied in order, where each step is a dictionary.

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
        - Options: `year`, `month`, `week`, `weekday`, `day`, `hour`, `minute`, `second`, `microsecond`
- `rotation`
    - `window_sizes`: a list of sizes for rolling median smoothing, classic baseline uses 100
- `sun`
    - `sun_features`: a list of sun features to include
      - Options: `elevation`, `azimuth`
- `add_holidays`
- `add_school_hours`
- `add_weather`
    - `weather_features`: a list of weather features to include
        - Options: `temp`, `dwpt`, `rhum`, `prcp`, `snow`, `wdir`, `wspd`, `wpgt`, `pres`, `tsun`, `coco`

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
        },
        {
            "kind": "sun",
            "sun_features": ["elevation", "azimuth"]
        },
        {
            "kind": "add_holidays"
        },
        {
            "kind": "add_school_hours"
        },
        {
            "kind": "add_weather",
            "weather_features": ["temp", "dwpt", "rhum", "prcp", "snow", "wdir", "wspd", "wpgt", "pres", "tsun", "coco"]
        }
]
```