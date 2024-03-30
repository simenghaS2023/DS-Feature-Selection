from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from tqdm import trange
import pandas as pd
from .iterative import IterativeFeatureSelector
from collections import deque

class GeneticAlgorithmFeatureSelector(IterativeFeatureSelector):
    def __init__(self, population_size = 20, n_generations = 100, crossover_rate = 0.8, mutation_rate = 0.1, n_estimators = 10, cv = 10, improvement_threshold = 0.005, delay_tolerance = 10):
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.improvement_threshold = improvement_threshold
        self.delay_tolerance = delay_tolerance
        self.fitness_history = None
        self.best_individual = None
        self.best_performance = None
        self.n_estimators = n_estimators
        self.cv = cv
        self.fitness_history = []
        super().__init__(n_estimators, cv, self.fitness_history)
        
    def select(self, X:pd.DataFrame, y:np.ndarray) -> pd.Index:
        self.fitness_history.clear()
        
        self.best_performance = 0.0
        last_improvement_time = 0
        best_average_fitness = 0.0
        
        population = self._create_population(X)
        for i in trange(self.n_generations):
            fitnesses = self._calculate_fitness_for_population(X, y, population)
            
            this_best_individual, this_best_fitness = max(zip(population, fitnesses), key=lambda x: x[1])
            if self.best_individual is None or this_best_fitness > self.best_performance:
                
                if this_best_fitness > self.best_performance * (1 + self.improvement_threshold):
                    last_improvement_time = i
                
                self.best_individual = this_best_individual
                self.best_performance = this_best_fitness

            this_mean_fitness = np.mean(fitnesses)
            
            
            self.fitness_history.append(this_mean_fitness)
            
            if i - last_improvement_time > self.delay_tolerance:
                break
            
            population = self._evolve(population, fitnesses, self.population_size, self.crossover_rate, self.mutation_rate)
            
        return X.columns[self.best_individual], i
        
    def _create_population(self, X):
        population = []
        for i in range(self.population_size):
            chromosome = np.random.choice([True, False], size=X.shape[1])
            population.append(chromosome)
        return np.array(population)
    
    def _calculate_fitness_for_population(self, X, y, population):
        fitnesses = [self._calculate_fitness(X, y, chromosome) for chromosome in population]
        return fitnesses
    
    def _calculate_fitness(self, X:pd.DataFrame, y, chromosome:np.ndarray):
        selected_columns = X.columns[chromosome]
        X_subset = X[selected_columns]
        
        rf = RandomForestClassifier(n_estimators=self.n_estimators, n_jobs=-1)
        accuracies = cross_val_score(rf, X_subset, y, cv=self.cv, scoring='accuracy', n_jobs=-1)
        return np.mean(accuracies)
    
    def _evolve(self, population, fitnesses, population_size, crossover_rate, mutation_rate):
        new_population = []
        
        # elitism
        elites_size = int((1-crossover_rate) * population_size)
        elites_idx = np.random.choice(population_size, size=elites_size, replace=False, p = fitnesses/np.sum(fitnesses))
        # elites = np.random.choice(population, size=elites_size, p=fitnesses/np.sum(fitnesses), replace=False)
        elites = population[elites_idx]
        new_population.extend(elites)
        
        # crossover
        crossover_size = (population_size - elites_size)
        for i in range(crossover_size):
            parents_idx = np.random.choice(population_size, size=2, replace=False, p=fitnesses/np.sum(fitnesses))
            parent1, parent2 = population[parents_idx]
            child = self._crossover(parent1, parent2)
            new_population.append(child)
            
        # mutation
        new_population = [self._mutate(chromosome, mutation_rate) for chromosome in new_population]
        
        return np.array(new_population)
    
    def _crossover(self, parent1, parent2):
        crossover_point = np.random.randint(0, len(parent1))
        child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        return child
    
    def _mutate(self, chromosome, mutation_rate):
        for i in range(len(chromosome)):
            if np.random.rand() < mutation_rate:
                chromosome[i] = not chromosome[i]
        return chromosome