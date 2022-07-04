from sklearn.svm import SVC
from skopt.space import Categorical, Integer, Real

from .base import BaseClassifier

class BestSVC(BaseClassifier):
    sklearn_estimator = SVC
    clf_param_grid = {
        'clf__C': Real(1e-2, 1e+3, 'log-uniform'),
        'clf__gamma': Real(1e-4, 1e+1, 'log-uniform'),
        'clf__degree': Integer(1, 3),
        'clf__kernel': Categorical(['linear', 'poly', 'rbf']),
        'clf__probability': Categorical([True]),
        'clf__class_weight': Categorical([None, 'balanced'])
    }
    n_iter = 50