"""
Configuration file for ML pipeline
Contains all constants, file paths, and hyperparameters
"""

import os

# File paths
DATA_PATH = "data/weatherHistory.csv"
MODEL_OUTPUT_PATH = "models/weather_best_models.pkl"
PLOTS_OUTPUT_DIR = "plots"

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)

# Data processing constants
DATE_COLUMN = 'Formatted Date'
TEMPERATURE_COLUMN = 'Temperature (C)'
PRECIPITATION_COLUMN = 'Precip Type'
MISSING_VALUE_THRESHOLD = 0.6  # Drop rows with less than 60% non-null values
MIN_CLASS_SAMPLES = 50  # Minimum samples required for a precipitation class

# Feature engineering
CLUSTER_FEATURES = ['hour', 'day_of_week', 'month']
N_CLUSTERS = 4

# Model training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 3

# Hyperparameter grids
RF_PARAM_GRID = {
    'n_estimators': [50, 100],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Initial model parameters (for quick training)
INITIAL_MODELS = {
    'classification': {
        'LogisticRegression': {'max_iter': 1000, 'random_state': RANDOM_STATE},
        'RandomForest': {
            'n_estimators': 20,
            'max_depth': 12,
            'random_state': RANDOM_STATE,
            'n_jobs': -1
        },
        'GradientBoosting': {
            'n_estimators': 10,
            'max_depth': 4,
            'random_state': RANDOM_STATE
        }
    },
    'regression': {
        'LinearRegression': {},
        'RandomForestReg': {
            'n_estimators': 20,
            'max_depth': 12,
            'random_state': RANDOM_STATE,
            'n_jobs': -1
        },
        'GradientBoostingReg': {
            'n_estimators': 10,
            'max_depth': 4,
            'random_state': RANDOM_STATE
        }
    }
}

# Visualization settings
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
PLOT_DPI = 300
FIGSIZE_LARGE = (15, 10)
FIGSIZE_MEDIUM = (10, 6)