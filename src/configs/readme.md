## Config options 

1. [Preprocessing steps](#preprocessing-steps)
2. [Preprocessing data location](#preprocessing-data-location)
3. [Feature engineering](#feature-engineering)
4. [Feature engineering data location](#feature-engineering-data-location)
5. [Models](#models)
6. [Ensemble](#ensemble)
7. [Loss](#loss)
8. [Hyperparameter optimization](#hyperparameter-optimization)
9. [Cross validation](#cross-validation)
10. [Scoring](#scoring)


### Preprocessing steps

These steps are executed in the order placed in the directionary

```
"preprocessing": ["pp1", "pp2"]
```

List of options:
- pp1
- pp2

### Preprocessing data location
<p>
Location out: Data created by preproccesing is stored in this location <br>
Location in: Data needed by preprocessing is stored in this location
</p>

```
"processed_loc_out": "./data/processed"
"processed_loc_in": "./data/raw"
```


### Feature engineering

Features that should be included during training and submission


``` 
"feature_engineering": ["fe1", "fe2"]
```

List of options
- fe1
- fe2

### Feature engineering data location
<p>
Location out: Data created by feature engineering is stored in this location <br>
Location in: Data needed by feature engineering is stored in this location
</p>

``` 
"fe_loc_out": "./data/features"
"fe_loc_in": "./data/processed"
```

### Models

A list of models and their specified configurations are included here. Multiple can be entered as this allows for the creation of ensembles. Additionally, the location they should be stored is included.
 
``` 
"models": {
    "model1name": {
        MODEL SPECIFIC CONFIG OPTIONS
    },
    "model2name": {
        MODEL SPECIFIC CONFIG OPTIONS
    }
}
```

#### Implemented Models types and config options

- example-fc-model
    - epochs (required)
    - loss (required)
    - optimizer (required)
    - lr
    - batch_size


Example of an example-fc-model configuration:

```
"ExampleModel": {
    "type": "example-fc-model",
    "epochs": 20,
    "batch_size": 32,
    "loss": "mae-torch",
    "optimizer": "adam-torch"
}
```

### Model store location

Specify location where models should be stored

```
"model_store_loc": "./tm",
```


### Ensemble

Ensemble specifications including the models used, the weight of each, and how the model predictions should be combined

```
"ensemble": {
    "models": ["model1name", "model2name"],
    "weights": [1, 2],
    "comb_method": "addition",
}
```

Combination methods
- Addition
- Max confidence

### Loss

Loss function to use for models

```
    "loss": "MSE"
```

Options
- MSE
- MAE

### Hyperparameter optimization

Specification for what to do for hyperparameter optimization

```
"hpo": {
    "apply": true | false,
    "method": "hpo1"
}
```

Options
- hpo1
- hpo2

### Cross validation

Choose specification for the cross validation and whether to do it or not

```
"cv": {
    "apply": true | false,
    "method": "stratifiedkfold
}
```

Cross validation options
- StratifiedKFold

### Scoring

Choose whether to do the scoring and show plots

```
"scoring": True | False
```