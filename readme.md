# Child Mind Institute - Detect Sleep States
## This repository contains the code for our solution to the detect sleep states competition

### Running main
Main should be run with the current working directory set to the directory main is in


### Config
The config.json file is used to set the paths of data folders, where to read the raw data and where to save our processed data, to enable or disbale logging to weights and biasses and using ensembles or doing hyperparameter optimization and cross validation. Below is an exampleconfig we used.

```JSON
{
    "name": "config",
    "is_kaggle": false,
    "log_to_wandb": true,
    "pred_with_cpu": false,
    "train_series_path": "data/raw/train_series.parquet",
    "train_events_path": "data/raw/train_events.csv",
    "test_series_path": "data/raw/test_series.parquet",
    "fe_loc_in": "data/processed",
    "processed_loc_out": "data/processed",
    "processed_loc_in": "data/raw",
    "model_store_loc": "tm",
    "model_config_loc": "model_configs",
    "ensemble": {
        "models": ["spectrogram-cnn-gru.json"],
        "weights": [1],
        "comb_method": "confidence_average",
        "pred_only": false
    },
    "cv": {
        "splitter": "group_k_fold",
        "scoring": ["score_full", "score_clean"],
        "splitter_params": {
            "n_splits": 5
        }
    },
    "train_for_submission": false,
    "scoring": true,
    "visualize_preds": {
        "n": 0,
        "browser_plot": false,
        "save": true
    }
}
```
### Logging and Kaggle
To log our experminet results we use weights and biasses. However since logging when running inference on kaggle does not make sense we have the is_kaggle and log_to_wandb as optional arguments in the config. For being able to run on kaggle in cpu notebook and to be able to use our GPUs locally we also have the pred with cpu argumnet in the config (using torch.device() would also use the cpu on kaggle if the GPU is disabled for the notebook so this is redundant but used).

### Model training
For training models set the config to ensemble and for models give a list of all model configs you would like to use and their weights. By giving multiple model configs which are located in the model_config_loc (specified in the config) folder and weigths, you can make ensembles and give each model a different weight for the confidence averaging step.

To do cross validation replace see the example config below

```JSON
{
    "name": "config",
    "is_kaggle": false,
    "log_to_wandb": true,
    "pred_with_cpu": false,
    "train_series_path": "data/raw/train_series.parquet",
    "train_events_path": "data/raw/train_events.csv",
    "test_series_path": "data/raw/test_series.parquet",
    "fe_loc_in": "data/processed",
    "processed_loc_out": "data/processed",
    "processed_loc_in": "data/raw",
    "model_store_loc": "tm",
    "model_config_loc": "model_configs",
    "hpo": "spectrogram-cnn-gru.json",
    "cv": {
        "splitter": "group_k_fold",
        "scoring": ["score_full", "score_clean"],
        "splitter_params": {
            "n_splits": 5
        }
    },
    "train_for_submission": false,
    "scoring": true,
    "visualize_preds": {
        "n": 0,
        "browser_plot": false,
        "save": true
    }
}
```

When the config is this way cross validation will be done using the parameters given with cv. The splitter, number of folds and what scores to calculate are all arguments (clean score refers to your score on data without NaNs). This config is also used when doing hpo with weights and biasses.

When train for submission is set to true the model will also be trained on the complete train set. The visualize preds arguments are used to generate plots using plotly or to simply save jpeg of our models predictions for each series compared to the real events.

To see how the preprocessing and feature engineering steps are chosen along with how hyperparameters for each model are set please refer to src/configs/readme.md and the individual model configs in the model_configs folder. Each model config lists all preprocessing, feature engineering steps along with model hyperparameters and the type of model in the formats defined in src/configs/readme.md. (There might be some methods that re not mentioned in the readme)

The training happens by passing all our data to the trainer class located in src/models/trainers.

### Model architectures
In our code the model classes (classes that have methods like train and pred) and model architectures are separated. To see our model architectures please refer to src/models/architectures and to see the model classes please refer to src/models.
