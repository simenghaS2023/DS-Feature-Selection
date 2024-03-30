import numpy as np
from tqdm import trange
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from .iterative import IterativeFeatureSelector

class SimulatedAnnealingFeatureSelector(IterativeFeatureSelector):
    def __init__(self, T0 = 1, cooling_rate = 0.99, final_temperature = 0.01, n_estimators = 10, cv = 25, improvement_threshold = 0.005, delay_tolerance = 200):
        self.T0 = T0
        self.cooling_rate = cooling_rate
        self.final_temperature = final_temperature
        self.improvement_threshold = improvement_threshold
        self.delay_tolerance = delay_tolerance
        self.best_individual = None
        self.best_performance = None
        self.performance_history = []
        super().__init__(n_estimators, cv, self.performance_history)
        
    def select(self, X, y):
        self.performance_history.clear()
        self.best_performance = 0.0
        temperature = self.T0
        current_individual = np.random.choice([True, False], size=X.shape[1])
        current_performance = self._calculate_performance(X, y, current_individual)
        
        last_improvement_time = 0
        
        expected_iterations = int(np.log(self.final_temperature/self.T0)/np.log(self.cooling_rate))
        
        for i in trange(expected_iterations + 1):
            if temperature < self.final_temperature:
                break
            new_individual = self._get_neighbor(current_individual)
            new_performance = self._calculate_performance(X, y, new_individual)
            
            # neighbor acceptance criterion
            if new_performance > current_performance or np.random.rand() < np.exp((new_performance - current_performance)/temperature):
                current_individual = new_individual
                current_performance = new_performance
                
            # early stopping control
            if current_performance > self.best_performance * (1 + self.improvement_threshold):
                last_improvement_time = i
            
            # update best individual
            if self.best_individual is None or current_performance > self.best_performance:
                self.best_individual = current_individual
                self.best_performance = current_performance
                
            self.performance_history.append(current_performance)
            
            if i - last_improvement_time > self.delay_tolerance:
                break
            
            temperature *= self.cooling_rate
        return X.columns[self.best_individual], i
            
    def _get_neighbor(self, individual):
        new_individual = individual.copy()
        idx = np.random.randint(0, len(individual))
        new_individual[idx] = not new_individual[idx]
        return new_individual
    
    def _calculate_performance(self, X, y, individual):
        selected_columns = X.columns[individual]
        X_subset = X[selected_columns]
        
        rf = RandomForestClassifier(n_estimators=self.n_estimators, n_jobs=-1)
        accuracies = cross_val_score(rf, X_subset, y, cv=self.cv, scoring='accuracy', n_jobs=-1)
        return np.mean(accuracies)