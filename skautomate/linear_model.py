from sklearn.linear_model import RidgeClassifier
from skopt.space import Categorical, Real

from .base import BaseClassifier

class BestRidgeClassifier(BaseClassifier):
    sklearn_estimator = RidgeClassifier
    clf_param_grid = {
        'clf__alpha': Real(1e-4, 1e+0, 'log-uniform'),
        'clf__class_weight': Categorical(['balanced'])
    }
    n_iter = 15