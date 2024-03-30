# Running the application

## Prerequisites
```python3``` must be installed on the system and available in command line.
## Via startup script
The startup script will create a virtual environment, install the dependencies and run the application. The script has been tested on Ubuntu 22.04 with bash and zsh. The script expects to be run from the root of the project directory.
### Steps
0. Add execute permissions to the script
    ```bash
    chmod +x run.sh
    ```
1. Run the startup script
    ```bash
    ./run.sh
    ```
## Manually
Execute the following steps from the root of the project directory.
### Steps
0. (optionally) Initialize a virtual environment
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
1. Install the dependencies
    ```bash
    pip install -r requirements.txt
    ```
2. Run the application
    ```bash
    streamlit run main.py
    ```

# Usage
By default, the application is served on http://localhost:8501. As the application is intended as a server, it is expected to run persistently and can be stopped by pressing `Ctrl+C` in the terminal.

With the default hyperparamaters, each algorithm should take around 1-2 minutes to run. The Genetic algorithm is the slowest, followed by the SKL algorithm and the Simulated Annealing algorithm is the fastest. A progress bar is displayed on the terminal to indicate the progress of the algorithm.

The application is a simple web application that allows the user to select a feature selection method and configure it with relevant hyperparameters. After receiving the hyperparameters, the application performs feature selection on the dataset and trains a model using the selected features. The user can then view the results of the model training and the selected features.

The application uses a subset of the SETAP process dataset. The data files used can be configured in ```app_config.json```. By default, the application used data files T1-T5, which are independent from one another. This can be changed to include other data files, although the application itself has not been tested with other data files.

Due to incompatible format between the Process and Product datasets, the application will not work with the Product dataset.

# Project structure
The project is structured as follows:
- ```main.py```: The main script that runs the Streamlit application.
- ```data/```: Contains the dataset used by the application.
- ```evaluation/```: Contains code for controlling the evaluation of the feature selection methods.
    - ```single_trial.py```: Contains the code to run a single trial of the feature selection method. It uses the given feature selection algorithm to obtain a subset of features, trains a new model using the selected features, and evaluates the model on a given test set.
    - ```multiple_trials.py```: Contains the code to run multiple trials of the feature selection method. It aggregates results from multiple trials and returns the average performance as well as the _popularity_ of each feature, which is the percentage of trials in which the feature was selected.
- ```feature_selection/```: Contains the classes for the feature selection methods.
    - ```genetic.py```: Contains the Genetic algorithm implementation.
    - ```simulated_annealing.py```: Contains the Simulated Annealing implementation.
    - ```skl.py```: Contains the Scikit-learn implementation based on RFECV.
- ```frontend/```: Contains the code to render the Streamlit application and to control the overall flow of the application.
- ```preprocessing/```: Contains the code to preprocess the dataset. The preprocessor loads the dataset, performs train-test split, and encode the label column.
- ```exp/```: Contains artifacts from the experiments conducted. It is not directly used by the application but some of the results are used in the report.
- ```app_config.json```: Contains the configuration for the application. It specifies the data files to be used by the application.
- ```requirements.txt```: Contains the dependencies required by the application.
- ```run.sh```: Contains the startup script to run the application.
- ```README.md```: Contains the documentation for the application.
- ```baseline.py```: Contains the code to run the baseline dummy algorithm which always selects all features.

