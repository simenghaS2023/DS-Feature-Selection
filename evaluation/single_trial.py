import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from tqdm import trange

class SingleTrialEvaluation:
    def __init__(self, selector_class):
        self.selector_class = selector_class
        
    def evaluate(self, X_train, y_train, X_test, y_test, *args, **kwargs):
        selector = self.selector_class(*args, **kwargs)
        selected_features, iterations = selector.select(X_train, y_train)
        
        clf = RandomForestClassifier(n_estimators=25, n_jobs=-1)
        clf.fit(X_train[selected_features], y_train)
        y_pred = clf.predict(X_test[selected_features])
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        n_features = len(selected_features)
        
        metrics = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'f1', "n_features", "iterations"])
        metrics.loc[0] = [accuracy, precision, recall, f1, n_features, iterations]
        
        return metrics, selected_features