### This is the directory for model json files

An example format:
```JSON
{
    "name": "111-Event-1D-Unet-CNN",
    "preprocessing": [
        {
            "kind": "mem_reduce",
            "id_encoding_path": "series_id_encoding.json"
        },
        {
            "kind": "similarity_nan",
            "as_feature": true
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
            "kind": "add_event_labels",
            "id_encoding_path": "series_id_encoding.json",
            "events_path": "data/raw/train_events.csv",
            "smoothing": 0
        },
        {
            "kind": "split_windows"
        }
    ],
    "feature_engineering": [
        {
            "kind": "time",
            "time_features": ["hour","minute"]
        },
        {
            "kind": "rotation",
            "window_sizes": [12, 36, 60, 90, 120, 150, 180, 240, 300, 360, 1000]
        }
    ],
    "pretraining": {
        "downsample": {
            "features": ["f_anglez", "f_enmo"],
            "methods": ["min", "max", "mean", "std", "median", "range", "var"],
            "standard": "mean"
        },
        "test_size": 0.2,
        "scaler": {
            "kind": "standard-scaler",
            "copy": true
        }
    },
    "cv": {
        "splitter": "group_shuffle_split",
        "scoring": ["score_full", "score_clean"],
        "splitter_params": {
            "n_splits": 1
        }
    },
    "architecture": {
        "type": "split-event-seg-unet-1d-cnn",
        "loss": "mse-torch",
        "optimizer": "adam-torch",
        "epochs": 52,
        "batch_size": 32,
        "lr": 0.0005,
        "hidden_layers": 16,
        "early_stopping": 7,
        "threshold": 0,
        "mask_unlabeled": true
    }
}
```