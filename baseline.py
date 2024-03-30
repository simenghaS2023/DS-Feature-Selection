from evaluation import MultipleTrialsEvaluation
from preprocessing import Preprocessor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from feature_selection import DummyFeatureSelector

preprocessor = Preprocessor()
X_train, y_train, X_test, y_test = preprocessor.preprocess()

evaluator = MultipleTrialsEvaluation(DummyFeatureSelector, trials=20)

metrics, feature_popularity = evaluator.evaluate(X_train, y_train, X_test, y_test)

metrics.to_csv("baseline_metrics.csv", index=True)

