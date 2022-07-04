from sklearn.neighbors import KNeighborsClassifier
from skopt.space import Categorical, Integer

from .base import BaseClassifier

class BestKNeighborsClassifier(BaseClassifier):
    sklearn_estimator = KNeighborsClassifier
    clf_param_grid = {
        'clf__n_neighbors': Integer(1, 20),
        'clf__weights': Categorical(['uniform', 'distance']),
        'clf__metric': Categorical(['euclidean', 'manhattan', 'minkowski']),
    }
    n_iter = 25