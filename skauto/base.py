import pickle
import warnings

from abc import ABC, abstractmethod
from typing import Type

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y, NotFittedError

from skopt import BayesSearchCV

class BaseClassifier(ABC):

    available_metrics = {
        'accuracy': accuracy_score,
        'balanced_accuracy': balanced_accuracy_score,
        'roc_auc': roc_auc_score,
        'f1': f1_score
    }

    prob_metrics = ['roc_auc']

    fs_param_grid = {}

    def __init__(self, score_method: str = 'accuracy') -> None:
        self._pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ("clf", self.sklearn_estimator())
        ])
        self._score_method = score_method
    
    def __str__(self) -> str:
        return str(self.sklearn_estimator.__name__)
    
    @property
    @abstractmethod
    def clf_param_grid(self) -> dict:
        pass

    @property
    @abstractmethod
    def n_iter(self) -> int:
        pass

    @property
    @abstractmethod
    def sklearn_estimator(self) -> Type[BaseEstimator]:
        pass

    @property
    def param_grid(self) -> dict:
        return self.fs_param_grid | self.clf_param_grid
    
    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline
    
    @pipeline.setter
    def pipeline(self, pipeline: Pipeline) -> None:
        if not isinstance(pipeline, Pipeline):
            raise TypeError
        self._pipeline = pipeline

    def check_is_fitted(self) -> None:
        if not self.is_fitted():
            raise NotFittedError("Object must be fitted before method call")
    
    def is_fitted(self) -> bool:
        try:
            check_is_fitted(self.grid_)
        except (AttributeError, NotFittedError):
            return False
        return True

    def fit(self, X: np.ndarray, y: np.ndarray, surpress_warnings = True, **kwargs) -> None:
        if y is None:
            check_array(X)
        else:
            check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        if self._score_method not in self.available_metrics.keys():
            raise ValueError(f"Scoring must be one of: {self.available_metrics.keys()}")
        self.grid_ = BayesSearchCV(
            self.pipeline,
            self.param_grid,
            n_jobs=-1,
            scoring=self._score_method,
            n_iter=self.n_iter,
            cv=5,
            **kwargs)
        print("Begin fitting best classifier for model: " + str(self))
        if surpress_warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.grid_.fit(X, y)
        else:
            self.grid_.fit(X, y)
        self.n_targets_ = len(np.unique(y))
        self.best_estimator_ = self.grid_.best_estimator_
        self.best_params_ = self.grid_.best_params_
        self.best_score_ = self.grid_.best_score_
        self.classes_ = self.grid_.classes_
        self.n_features_in_ = self.grid_.n_features_in_
        print("Done!")
        print(f"{self._score_method}: {self.best_score_}")

    def predict(self, X: np.ndarray) -> None:
        self.check_is_fitted()
        check_array(X)
        return self.grid_.predict(X)
    
    def get_y_score(self, X: np.ndarray):
        self.check_is_fitted()
        check_array(X)
        if len(self.classes_) != 2:
            raise ValueError("This method was created for binary classes")
        try:
            return self.grid_.predict_proba(X)[:, 1]
        except AttributeError:
            return self.grid_.decision_function(X)

    def score(self, X: np.ndarray, y: np.ndarray):
        return self.grid_.score(X, y)
    
    def save(self, filepath):
        with open(filepath, "wb") as file:
            pickle.dump(self, file)