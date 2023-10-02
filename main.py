# This file does the training of the model

# Imports
import pandas as pd
from src.configs.load_config import ConfigLoader
import submit_to_kaggle
import wandb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load config file
config = None


def main(config: ConfigLoader):
    # Initialize wandb
    if config.get_log_to_wandb():
        # Initialize wandb
        wandb.init(
            project='detect-sleep-states',

            config=config.get_config()
        )
    else:
        print("Not logging to wandb.")

    # Do training here
    df = pd.read_parquet(config.get_pp_in() + "/test_series.parquet")

    # Initialize preprocessing steps
    print("-------- PREPROCESSING ----------")
    pp_steps, pp_s = config.get_pp_steps()
    processed = df
    # Get the preprocessing steps as a list of str to make the paths
    for i, step in enumerate(pp_steps):
        # Passes the current list because its needed to write to if the path doesnt exist
        processed = step.run(processed, pp_s[:i + 1])

    # Initialize feature engineering steps
    print("-------- FEATURE ENGINEERING ----------")
    fe_steps, fe_s = config.get_features()
    featured_data = processed
    for i, fe_step in enumerate(fe_steps):
        # Also pass the preprocessing steps to the feature engineering step
        # to save fe for each possible pp combination
        featured_data = fe_steps[fe_step].run(processed, fe_s[:i + 1], pp_s)

    # Pretrain processstep (splitting data into train and test, standardization, etc.)
    print("-------- PRETRAINING ----------")
    pretrain = config.get_pretraining()

    # Standardize data
    exclude_columns = ['series_id', 'timestamp',
                       'window', 'step', 'awake']  # 'onset', 'wakeup']
    scaler = None
    if pretrain["standardize"] == "standard":
        scaler = StandardScaler()
    elif pretrain["standardize"] == "minmax":
        scaler = MinMaxScaler()

    # Drop columns to exclude from standardization
    data_to_standardize = featured_data.drop(columns=exclude_columns)
    data_to_standardize = scaler.fit_transform(data_to_standardize)
    # Add columns back to standardized data
    standardized_data = pd.DataFrame(
        data_to_standardize, columns=featured_data.columns.drop(exclude_columns))
    featured_data = pd.concat(
        [featured_data[exclude_columns], standardized_data], axis=1)
    print("Data standardized")

    # Train test split on series id
    # Check if test size key exists in pretrain
    train_data = None
    test_data = None
    if "test_size" in pretrain:
        groups = featured_data["series_id"]
        gss = GroupShuffleSplit(
            n_splits=1, test_size=pretrain["test_size"], random_state=42)
        train_idx, test_idx = next(gss.split(featured_data, groups=groups))
        train_data = featured_data.iloc[train_idx]
        test_data = featured_data.iloc[test_idx]
        print("Data split into train and test")
    else:
        cv = config.get_cv()
        print("Crossvalidation will be used instead of train test split")

    # Initialize models
    print("-------- TRAINING MODELS ----------")
    models = config.get_models()
    for model in models:
        models[model].train(train_data)

    # Get saved models directory from config
    store_location = config.get_model_store_loc()

    # TODO Add crossvalidation to models #107
    print("-------- ENSEMBLING ----------")
    ensemble = config.get_ensemble(models)

    # TODO ADD preprocessing of data suitable for predictions #103

    ensemble.pred(test_data)

    # Initialize loss
    # TODO assert that every model has a loss function #67

    # TODO Hyperparameter optimization for ensembles #101
    hpo = config.get_hpo()
    hpo.optimize()

    # Initialize CV
    cv = config.get_cv()
    cv.run()

    # Train model fully on all data
    # TODO Mark best model from CV/HPO and train it on all data
    if config.get_train_for_submission():
        best_model = None
        best_model.train(featured_data)
        # Add submit in name for saving
        best_model.save(store_location + "/submit_" + best_model.name + ".pt")

    # Get scoring
    scoring = config.get_scoring()
    if scoring:
        # Do scoring
        pass

    # TODO Add Weights and biases to model training and record loss and acc #106

    # TODO ADD scoring to WANDB #108

    # [optional] finish the wandb run, necessary in notebooks
    if config.get_log_to_wandb():
        wandb.finish()


if __name__ == "__main__":
    # Load config file
    config = ConfigLoader("config.json")

    # Train model
    main(config)

    # Create submission
    submit_to_kaggle.submit(config.get_pp_in() + "/test_series.parquet", False)
