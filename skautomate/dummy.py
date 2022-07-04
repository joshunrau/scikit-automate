from sklearn.dummy import DummyClassifier
from skopt.space import Categorical

from .base import BaseClassifier

class BestDummyClassifier(BaseClassifier):
    sklearn_estimator = DummyClassifier
    clf_param_grid = {
        'clf__strategy': Categorical(['most_frequent'])
    }
    n_iter = 1