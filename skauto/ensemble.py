from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from skopt.space import Categorical, Integer, Real

from .base import BaseClassifier

class BestRandomForestClassifier(BaseClassifier):
    sklearn_estimator = RandomForestClassifier
    clf_param_grid = {
        'clf__n_estimators': Integer(50, 500),
        'clf__max_depth': Integer(5, 50),
        'clf__max_features': Real(1e-2, 1e+0, 'log-uniform'),
        'clf__min_samples_split': Integer(2, 5),
        'clf__min_samples_leaf': Integer(1, 5),
        'clf__class_weight': Categorical(['balanced'])
    }
    n_iter = 40

class BestGradientBoostingClassifier(BaseClassifier):
    sklearn_estimator = GradientBoostingClassifier
    clf_param_grid = {
        "clf__loss": Categorical(["log_loss"]),
        "clf__learning_rate": Real(1e-3, 5e-1, 'log-uniform'),
        "clf__min_samples_split": Real(0.1, 0.9, 'log-uniform'),
        "clf__min_samples_leaf": Real(0.1, 0.5, 'log-uniform'),
        "clf__max_depth": Integer(2, 10),
        "clf__max_features": Categorical(["log2","sqrt"]),
        "clf__criterion": Categorical(["friedman_mse",  "squared_error"]),
        "clf__subsample": Real(0.5, 1, 'log-uniform')
    }
    n_iter = 50