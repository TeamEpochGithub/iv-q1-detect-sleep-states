# Preprocessing

This directory contains the preprocessing steps. The following steps are currently implemented:

- `add-noise`
    - Adds gaussian noise to the sensor data.
- `add_state_labels`
    - Labels the data in a way that each timestep gets a label.
        - `0`: asleep.
        - `1`: awake.
        - `2`: `NaN`, not labeled.
- `split_windows` (`start_hour`: `int`, `window_size`: `int`)
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
