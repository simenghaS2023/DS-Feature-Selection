import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class FeatureSelector:
    def __init__(self, n_estimators=25, cv=10):
        self.n_estimators = n_estimators
        self.cv = cv
        
    def select(self, X:pd.DataFrame, y:np.ndarray) -> pd.Index:
        raise NotImplementedError("Subclasses must implement this method")
    
    def evaluate(self, X:pd.DataFrame, y:np.ndarray, trials=10):
        selected_features = self.select(X, y)
        return self._evaluate_subset(X, y, selected_features, trials)
    
    def _evaluate_subset(self, X:pd.DataFrame, y:np.ndarray, selected_features:pd.Index, trials:int):
        X_subset = X[selected_features]
        clfs = [RandomForestClassifier(n_estimators=self.n_estimators, n_jobs=-1) for _ in range(trials)]
        predictions = np.zeros((len(X), trials))
        for i, clf in enumerate(clfs):
            predictions[:, i] = clf.fit(X_subset, y).predict(X_subset)
        
        