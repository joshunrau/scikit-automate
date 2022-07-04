from skauto.base import BaseClassifier
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from skopt.space import Real

class BestLogisticRegression(BaseClassifier):
  sklearn_estimator = LogisticRegression
  clf_param_grid = {
    'clf__C': Real(1e-2, 1e+2, 'log-uniform'),
    }
  n_iter = 10

X, y = load_iris(return_X_y=True)
clf = BestLogisticRegression()
clf.fit(X, y)