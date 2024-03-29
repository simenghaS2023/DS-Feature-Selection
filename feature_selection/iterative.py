from .base import FeatureSelector
import matplotlib.pyplot as plt

class IterativeFeatureSelector(FeatureSelector):
    def __init__(self, n_estimators=25, cv=10, perforamnce_history = []):
        super().__init__(n_estimators, cv)
        
    def select(self, X, y):
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_history(self):
        return self.performance_history
    
    def get_best(self):
        return self.best_individual, self.best_performance
    
    def plot_history(self, plot_title = 'Performance History'):
        plt.plot(self.performance_history)
        plt.title(plot_title)
        plt.xlabel('Iteration')
        plt.ylabel('Performance')
        plt.show()