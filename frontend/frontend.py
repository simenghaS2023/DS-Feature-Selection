import numpy as np
import streamlit as st
from preprocessing import Preprocessor
from feature_selection import SimulatedAnnealingFeatureSelector, SKLFeatureSelector, GeneticAlgorithmFeatureSelector, FeatureSelector, IterativeFeatureSelector
from evaluation import MultipleTrialsEvaluation
import matplotlib.pyplot as plt

# def main():
#     preprocessor = Preprocessor()
#     train_X, train_y, test_X, test_y = preprocessor.preprocess()

#     train_X.info()
#     test_X.info()

#     # selector = SimulatedAnnealingFeatureSelector(cv=2)
#     # selected_features = selector.select(train_X, train_y)
#     # print(selected_features)

#     evaluation = MultipleTrialsEvaluation(SKLFeatureSelector, trials=5)
#     metrics, feature_popularity = evaluation.evaluate(train_X, train_y, test_X, test_y, cv=2)

#     print(metrics)
#     print(feature_popularity)


# if __name__ == "__main__":
#     main()

FEATURE_SELECTOR_NAME_TO_CLASS = {
    "SKLearn RFECV": SKLFeatureSelector,
    "Genetic Algorithm": GeneticAlgorithmFeatureSelector,
    "Simulated Annealing": SimulatedAnnealingFeatureSelector
}

class Frontend:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.train_X, self.train_y, self.test_X, self.test_y = self.preprocessor.preprocess()
        
    def run(self):
        st.write("# Feature Selection")
        st.text(
            "All algorithms provided use Random Forest classifiers as the base estimator.")
        st.text(
            "Hover over the question mark icons for more information on each hyperparameter.")
        feature_selector_choice, algorithm_hyperparameters, early_stop_hyperparameters, multiple_trials_hyperparameters = self.configure() or (None, None, None, None)
        if feature_selector_choice is None:
            return None
        if st.button("Run"):
            evaluation = MultipleTrialsEvaluation(FEATURE_SELECTOR_NAME_TO_CLASS[feature_selector_choice], **multiple_trials_hyperparameters)
            with st.spinner("Running..."):
                metrics, feature_popularity = evaluation.evaluate(self.train_X, self.train_y, self.test_X, self.test_y, **algorithm_hyperparameters, **early_stop_hyperparameters)
                # metrics, feature_popularity, selected_features = evaluation.evaluate(self.train_X, self.train_y, self.test_X, self.test_y, **algorithm_hyperparameters, **early_stop_hyperparameters)
            
            st.write("## Metrics")
            st.dataframe(metrics)
            st.write("## Feature Popularity")
            st.write(feature_popularity)
            
        
        
    def _prompt_algorithm_selection(self):
        # feature_selector_choice = st.selectbox("Feature Selector", [
        #     "Choose a feature selection algorithm", "SKLearn RFECV", "Genetic Algorithm", "Simulated Annealing"], index=2)
        feature_selector_choice = st.selectbox("Feature Selector", [
            "Choose a feature selection algorithm", *list(FEATURE_SELECTOR_NAME_TO_CLASS.keys())], index=0)
        if feature_selector_choice == "Choose a feature selection algorithm":
            return None
        # st.write("## Common hyperparameters:")
        # n_estimators = st.slider(
        #     "Number of estimators", min_value=1, max_value=25, value=2)
        # st.markdown("> Number of trees used by the Random Forest classifier")
        # cv = st.slider("Cross Validation", min_value=2, max_value=10, value=2)
        # st.markdown("> Number of folds used to calculate training accuracies")
        
        return feature_selector_choice
        # return feature_selector_choice, n_estimators, cv
    
    def _get_hyperparameters(self, feature_selector_choice):
        
        algorithm_hyperparameters = {}
        early_stop_hyperparameters = {}
        multiple_trials_hyperparameters = {}
        
        st.write("## Common hyperparameters")
        n_estimators = st.slider(
            "Number of estimators", min_value=1, max_value=25, value=2, help="Number of trees used by the Random Forest classifier, increase for more complex but slower models")
        cv = st.slider("Cross Validation", min_value=2, max_value=10, value=2, help="Number of folds used to calculate training accuracies, increase for more precise estimates at the expense of speed")
        
        algorithm_hyperparameters["n_estimators"] = n_estimators
        algorithm_hyperparameters["cv"] = cv

        
        if feature_selector_choice == "SKLearn RFECV":
            pass
        else:
            st.write(f"## Additional hyperparameters for {feature_selector_choice}")
        
            if feature_selector_choice == "Genetic Algorithm":
                population_size = st.slider(
                    "Population Size", min_value=5, max_value=20, value=10, help="Number of individuals in the population, increase for more diverse solutions at the expense of speed")
                n_generations = st.slider(
                    "Generations", min_value=1, max_value=200, value=10, help="Maximum number of generations to run the algorithm. Increase for better solutions at the expense of speed")
                mutation_rate = st.slider(
                    "Mutation Rate", min_value=0.0, max_value=1.0, value=0.1, help="Probability of a gene being mutated, increase for more exploration and less exploitation")
                crossover_rate = st.slider(
                    "Crossover Rate", min_value=0.0, max_value=1.0, value=0.9, help="Ratio of individuals produced by crossover, increase for more exploration and less exploitation")
                estimated_iterations = n_generations
                
                algorithm_hyperparameters["population_size"] = population_size
                algorithm_hyperparameters["n_generations"] = n_generations
                algorithm_hyperparameters["mutation_rate"] = mutation_rate
                algorithm_hyperparameters["crossover_rate"] = crossover_rate
                
                
            elif feature_selector_choice == "Simulated Annealing":
                T0e = st.slider("Initial Temperature Exponent &e", min_value=-10,
                            max_value=10, value=0, help="Exponent for the initial temperature of the annealing process. Actual value is 2^e. Increase for more exploration and less exploitation")
                T0 = np.power(2.0, T0e)
                cooling_rate = st.slider(
                    "Cooling Rate", min_value=0.9, max_value=1.0, value=0.99, help="Rate at which the temperature decreases, increase for faster convergence and the expense of exploration")
                final_temperature = st.slider(
                    "Final Temperature", min_value=0.01, max_value=1.0, value=0.01, help="Temperature at which the annealing process stops, increase for potentially better solutions at the expense of speed")
                estimated_iterations = int(np.log(final_temperature/T0)/np.log(cooling_rate))
                
                algorithm_hyperparameters["T0"] = T0
                algorithm_hyperparameters["cooling_rate"] = cooling_rate
                algorithm_hyperparameters["final_temperature"] = final_temperature
                
            st.write("## Early stop hyperparameters")
            improvement_threshold = st.slider(
                    "Improvement Threshold", min_value=0.00, max_value=0.1, value=0.005, step=0.001, format="%.3f", help="Percentage of improvement required to be considered progress-making, higher values require more significant improvements to continue")
            delay_tolerance = st.slider(
                "Delay Tolerance", min_value=1, max_value=estimated_iterations, value=10, help="Number of iterations without progress before the algorithm stops, higher values allow for more non-progress iterations before stopping")
            early_stop_hyperparameters["improvement_threshold"] = improvement_threshold
            early_stop_hyperparameters["delay_tolerance"] = delay_tolerance
            
        st.write("## Multiple-trial hyperparameters")
        trials = st.slider("Number of Trials", min_value=1, max_value=20, value=5, help="Number of times to run the algorithm (to reduce variance), increase for more stable results at the expense of speed")
        multiple_trials_hyperparameters["trials"] = trials
        
        return algorithm_hyperparameters, early_stop_hyperparameters, multiple_trials_hyperparameters
            
        
        
        
    def configure(self):
        feature_selector_choice = self._prompt_algorithm_selection()
        if feature_selector_choice is None:
            return None
        algorithm_hyperparameters, early_stop_hyperparameters, multiple_trials_hyperparameters = self._get_hyperparameters(feature_selector_choice)
        return feature_selector_choice, algorithm_hyperparameters, early_stop_hyperparameters, multiple_trials_hyperparameters
        
