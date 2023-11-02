# Config variables
pred_with_cpu: bool = False  # Whether to use CPU or GPU for prediction
window_size: int = 17280  # The size of the window in steps. Default is 24 * 60 * 12 = 17280
downsampling_factor: int = 0  # The factor to downsample by
stage: str = "load_config"  # The stage of the pipeline
substage: str = "set_globals"  # The substage of the pipeline

# Info about the data
X_columns: dict[str, int] = {}  # The names of the features
y_columns: dict[str, int] = {}  # The names of the labels

# Cross Validation data
cv_current_fold: int = 0  # The current fold of the cross validation
