# Config variables
pred_with_cpu: bool = False  # Whether to use cpu or gpu
window_size: int = 17280  # The size of the window in steps. Default is 24 * 60 * 12 = 17280
downsampling_factor: int = 0  # The factor to downsample by

# Info about the data
# train_n: int = 8000  # The number of training series
# test_n: int = 2000  # The number of testing series
# X_train_shape: tuple = (train_n, window_size, 3)
X_columns: dict[str, int] = {}
# y_train_shape: tuple = (train_n, window_size, 1)
y_columns: dict[str, int] = {}
# X_test_shape: tuple = (test_n, window_size, 3)
# X_test_columns: dict[str, int] = {'enmo': 0, 'anglez': 1}
# y_test_shape: tuple = (test_n, window_size, 1)
# y_test_columns: dict[str, int] = {'awake': 0}

# Cross Validation data
cv_current_fold: int = 0
