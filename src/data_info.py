# Config variables
pred_with_cpu: bool = False  # Whether to use CPU or GPU for prediction
window_size_before: int = 17280  # The size of the window in steps before donwsampling. Default is 24 * 60 * 12 = 17280
window_size: int = 17280  # The size of the window in steps, potentially after downsampling. Default is 24 * 60 * 12 = 17280
downsampling_factor: int = 1  # The factor to downsample by
stage: str = "load_config"  # The stage of the pipeline
substage: str = "set_globals"  # The substage of the pipeline
plot_summary: bool = False  # Whether to plot the summary of the model


scorings: list[str] = ["score_full", "score_clean"]  # The scorings to use for evaluation

# Data info variables
latitude = 40.730610  # The latitude of the data of NYC
longitude = -73.935242  # The longitude of the data of NYC

# Info about the data
X_columns: dict[str, int] = {}  # The names of the features
y_columns: dict[str, int] = {}  # The names of the labels

# Cross Validation data
cv_current_fold: int = 0  # The current fold of the cross validation
cv_unique_series: list[str] = []
