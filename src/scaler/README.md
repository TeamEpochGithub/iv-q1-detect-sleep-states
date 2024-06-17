# Scalers

This contains the different scalers used for data standardization. The following (`sklearn`) scalers are implemented:

- "standard-scaler" for `StandardScaler`
    - Optional parameters `copy`, `with_mean`, `with_std`
- "minmax-scaler" for `MinMaxScaler`
    - Optional parameters `feature_range`, `copy`, `clip`
- "robust-scaler" for `RobustScaler`
    - Optional parameters `with_centering`, `with_scaling`, `quantile_range`, `copy`, `unit_variance`
- "maxabs-scaler" for `MaxAbsScaler`
    - Optional parameter `copy`
- "quantile-transformer" for `QuantileTransformer`
    - Optional
      parameters `n_quantiles`, `output_distribution`, `ignore_implicit_zeros`, `subsample`, `random_state`, `copy`
- "power-transformer" for `PowerTransformer`
    - Optional parameters `method`, `standardize`, `copy`
- "normalizer" for `Normalizer`
    - Optional parameters `norm`, `copy`
- "none" for no scaling

For more info about the different scalers and their parameters,
see [here](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing) and
[here](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#plot-all-scaling-standard-scaler-section).
