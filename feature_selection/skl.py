from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from .base import FeatureSelector

class SKLFeatureSelector(FeatureSelector):
    def __init__(self, n_estimators=25, cv=10):
        self.n_estimators = n_estimators
        self.cv = cv
        
    def select(self, X, y):
        clf = RandomForestClassifier(n_estimators=self.n_estimators, n_jobs=-1)
        selector = RFECV(clf, cv=self.cv, scoring='accuracy', n_jobs=-1)
        selector.fit(X, y)
        
        return X.columns[selector.support_], 0