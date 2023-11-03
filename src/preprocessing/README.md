# Preprocessing
This directory contains the preprocessing steps. 
These steps are implemented as classes that inherit from the `PP` class in `src/preprocessing/pp.py`.
These classes apply various preprocessing steps to the data to prepare it for the models. 
The following steps are currently implemented:

- `mem_reduce`
    - Parameters: `id_encoding_path: Optional[str] = None`
    - Reduces the memory usage of the dataframe. Encodes the series IDs to unique ints and converts the timestamp to
      a datetime object.
- `add-noise`
    - Adds gaussian noise to the sensor data.
- `add_state_labels`
    - Parameters: `id_encoding_path: str`, `events_path: str`
    - Labels the data in a way that each timestep gets a label.
        - `0`: asleep.
        - `1`: awake.
        - `2`: `NaN`, not labeled.
- `split_windows`
    - Parameters: `start_hour: int = 15`
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
        - 2: `hot-NaN`
        - 3: `hot-unlabeled`

Example:
```JSON
{
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
      "start_hour": 15
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
}
```