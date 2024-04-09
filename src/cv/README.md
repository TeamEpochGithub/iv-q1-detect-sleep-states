# Cross Validation

This directory contains the code for the cross validation of the models.

## Splitters

A splitter splits the data into train and test sets.
The splitters for groups are likely most relevant for our data.

List of splitter options with parameters (
see [sklearn.model_selection](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection) for
more details):

- `group_k_fold`
    - `n_splits: int`, default = 5
- `group_shuffle_split`
    - `n_splits: int`, default = 5
    - `test_size: float`, default = 0.2
    - `train_size: float | int`, default = None
    - `random_state: int`, default = None
- `k_fold`
    - `n_splits: int`, default = 5
    - `shuffle: bool`, default = False
    - `random_state: int`, default = None
- `leave_one_group_out`
- `leave_p_group_out`
    - `n_groups: int`, default = 2
- `leave_one_out`
- `leave_p_out`
    - `p: int`
- `predefined_split`
    - `test_fold: array-like` of shape `(n_samples,)`
- `shuffle_split`
    - `n_splits: int`, default = 5
    - `test_size: float`, default = 0.2
    - `train_size: float | int`, default = None
    - `random_state: int`, default = None
- `stratified_k_fold`
    - `n_splits: int`, default = 5
    - `shuffle: bool`, default = False
    - `random_state: int`, default = None
- `stratified_shuffle_split`
    - `n_splits: int`, default = 10
    - `test_size: float | int`, default = None
    - `train_size: float | int`, default = None
    - `random_state: int`, default = None
- `stratified_group_k_fold`
    - `n_splits: int`, default = 5
    - `shuffle: bool`, default = False
    - `random_state: int`, default = None
- `time_series_split`
    - `n_splits: int`, default = 5
    - `max_train_size: int`, default = None
    - `test_size: int`, default = None
    - `gap: int`, default = 0

## Scorers

List of scorer options:

- `score_full`: computes the score for the full dataset
- `score_clean` computes the score for the clean dataset

Currently, the scorers have only been tested with the `seg-unet-1d-cnn`.

## Usage

Example:

```JSON
"cv": {
    "splitter": "group_shuffle_split",
    "scoring": ["score_full", "score_clean"],
    "splitter_params": {
        "n_splits": 1,
        "test_size": 0.2
    }
}
```