import pandas as pd
from tqdm import trange
from .single_trial import SingleTrialEvaluation

# class MultipleTrialsFeatureSelector(FeatureSelector):
#     def __init__(self, selector_class, trials=10):
#         self.selector_class = selector_class
#         self.trials = trials
        
#     def select(self, train_X, train_y, *args, **kwargs):
#         best_score = 0
#         best_features = []
#         for i in range(self.trials):
#             selector = self.selector_class(*args, **kwargs)
#             selected_features = selector.select(train_X, train_y)

# class MultipleTrialsEvaluation:
#     def __init__(self, selector_class, trials=10):
#         self.selector_class = selector_class
#         self.trials = trials
        
#     def evaluate(self, X_train, y_train, X_test, y_test, *args, **kwargs):
#         metrics = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'f1', "n_features"])
#         selected_features_per_trial = pd.DataFrame(columns=X_train.columns, index=range(self.trials), data = 0)
#         for i in trange(self.trials):
#             selector = self.selector_class(*args, **kwargs)
#             selected_features = selector.select(X_train, y_train)
            
#             selected_features_per_trial.loc[i, selected_features] = 1
            
#             clf = RandomForestClassifier(n_estimators=25, n_jobs=-1)
#             clf.fit(X_train[selected_features], y_train)
#             y_pred = clf.predict(X_test[selected_features])
            
#             accuracy = accuracy_score(y_test, y_pred)
#             precision = precision_score(y_test, y_pred)
#             recall = recall_score(y_test, y_pred)
#             f1 = f1_score(y_test, y_pred)
#             n_features = len(selected_features)
            
#             metrics.loc[i] = [accuracy, precision, recall, f1, n_features]
            
#         feature_popularity = selected_features_per_trial.sum()
#         feature_popularity = feature_popularity / self.trials
#         feature_popularity = feature_popularity.sort_values(ascending=False)
        
#         return metrics, feature_popularity
            
class MultipleTrialsEvaluation:
    def __init__(self, selector_class, trials=10):
        self.selector_class = selector_class
        self.trials = trials
        
    def evaluate(self, X_train, y_train, X_test, y_test, *args, **kwargs):
        selected_features = pd.DataFrame(columns=X_train.columns, index=range(self.trials), data=0)
        metrics_array = []
        for i in trange(self.trials):
            trial = SingleTrialEvaluation(self.selector_class)
            metrics, this_selected_features = trial.evaluate(X_train, y_train, X_test, y_test, *args, **kwargs)
            metrics_array.append(metrics)
            selected_features.loc[i, this_selected_features] = 1
            
        selected_features.set_index(pd.Index(range(1, self.trials+1), name="Trial"), inplace=True)
        n_features_per_trial = selected_features.sum(axis=1)
        metrics = pd.concat(metrics_array).set_index(pd.Index(range(1, self.trials+1), name="Trial"))
        metrics['n_features'] = n_features_per_trial
        
        feature_popularity = (selected_features.sum()/self.trials).sort_values(ascending=False)
        feature_popularity = feature_popularity.rename('popularity')
        
        # add average to last row
        metrics.loc['average'] = metrics.mean()
        
        
        
        return metrics, feature_popularity
                
                
            
        
                
            
            
            
