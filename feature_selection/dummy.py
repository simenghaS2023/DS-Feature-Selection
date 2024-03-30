import numpy as np
from tqdm import trange
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from .base import FeatureSelector

class DummyFeatureSelector(FeatureSelector):
    def __init__(self, n_estimators = 10, cv = 25):
        self.n_estimators = n_estimators
        self.cv = cv
        super().__init__(n_estimators, cv)
        
    def select(self, X, y):
        return X.columns, 0
    