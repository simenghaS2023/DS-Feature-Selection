from preprocessing import Preprocessor
from feature_selection import SimulatedAnnealingFeatureSelector, SKLFeatureSelector, GeneticAlgorithmFeatureSelector, FeatureSelector, IterativeFeatureSelector
from evaluation import MultipleTrialsEvaluation
import matplotlib.pyplot as plt
from frontend import Frontend

def main():
    preprocessor = Preprocessor()
    Frontend(preprocessor).run()
    
if __name__ == "__main__":
    main()