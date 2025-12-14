import os
import json
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Path to save best hyperparameters
BEST_PARAMS_FILE = "models/best_rf_params.json"
os.makedirs(os.path.dirname(BEST_PARAMS_FILE), exist_ok=True)


def tune_or_load_rf_classifier(X_train, y_train, param_grid, cv=3, scoring='f1_macro', random_state=42):
    """Tune RF classifier or load existing best parameters"""
    if os.path.exists(BEST_PARAMS_FILE):
        with open(BEST_PARAMS_FILE, 'r') as f:
            params_data = json.load(f)
        best_params = params_data.get('rf_classifier')
        if best_params:
            msg = f"✅ Loaded saved RF classifier hyperparameters: {best_params}"
            print(msg)
            logger.info(msg)
            model = RandomForestClassifier(**best_params, random_state=random_state, n_jobs=-1)
            model.fit(X_train, y_train)
            return {'best_model': model, 'best_params': best_params}

    # If params do not exist, tune
    print("🔄 Tuning Random Forest Classifier...")
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    rf_grid.fit(X_train, y_train)
    best_params = rf_grid.best_params_
    msg = f"✅ Best RF Classifier params: {best_params}, CV {scoring}: {rf_grid.best_score_:.4f}"
    print(msg)
    logger.info(msg)

    # Save to JSON
    params_data = {}
    if os.path.exists(BEST_PARAMS_FILE):
        with open(BEST_PARAMS_FILE, 'r') as f:
            params_data = json.load(f)
    params_data['rf_classifier'] = best_params
    with open(BEST_PARAMS_FILE, 'w') as f:
        json.dump(params_data, f, indent=4)

    return {'best_model': rf_grid.best_estimator_, 'best_params': best_params}


def tune_or_load_rf_regressor(X_train, y_train, param_grid, cv=3, scoring='r2', random_state=42):
    """Tune RF regressor or load existing best parameters"""
    if os.path.exists(BEST_PARAMS_FILE):
        with open(BEST_PARAMS_FILE, 'r') as f:
            params_data = json.load(f)
        best_params = params_data.get('rf_regressor')
        if best_params:
            msg = f"✅ Loaded saved RF regressor hyperparameters: {best_params}"
            print(msg)
            logger.info(msg)
            model = RandomForestRegressor(**best_params, random_state=random_state, n_jobs=-1)
            model.fit(X_train, y_train)
            return {'best_model': model, 'best_params': best_params}

    # If params do not exist, tune
    print("🔄 Tuning Random Forest Regressor...")
    rf_grid = GridSearchCV(
        RandomForestRegressor(random_state=random_state, n_jobs=-1),
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    rf_grid.fit(X_train, y_train)
    best_params = rf_grid.best_params_
    msg = f"✅ Best RF Regressor params: {best_params}, CV {scoring}: {rf_grid.best_score_:.4f}"
    print(msg)
    logger.info(msg)

    # Save to JSON
    params_data = {}
    if os.path.exists(BEST_PARAMS_FILE):
        with open(BEST_PARAMS_FILE, 'r') as f:
            params_data = json.load(f)
    params_data['rf_regressor'] = best_params
    with open(BEST_PARAMS_FILE, 'w') as f:
        json.dump(params_data, f, indent=4)

    return {'best_model': rf_grid.best_estimator_, 'best_params': best_params}


def train_all_models(preprocessed_data: dict, param_grid: dict, cv: int = 3, random_state: int = 42) -> dict:
    """
    Train all models, using saved hyperparameters if available.
    Prints progress to terminal.
    """
    results = {}

    # Classification
    X_clf = preprocessed_data['classification']['X_train']
    y_clf = preprocessed_data['classification']['y_train']
    print("\n=== Training Random Forest Classifier ===")
    clf_result = tune_or_load_rf_classifier(X_clf, y_clf, param_grid, cv=cv, random_state=random_state)
    results['best_classifier'] = clf_result['best_model']
    results['clf_tuning_results'] = clf_result

    # Regression
    X_reg = preprocessed_data['regression']['X_train']
    y_reg = preprocessed_data['regression']['y_train']
    print("\n=== Training Random Forest Regressor ===")
    reg_result = tune_or_load_rf_regressor(X_reg, y_reg, param_grid, cv=cv, random_state=random_state)
    results['best_regressor'] = reg_result['best_model']
    results['reg_tuning_results'] = reg_result

    # Add empty initial models for compatibility with evaluate.py
    results['initial_classifiers'] = {}
    results['initial_regressors'] = {}

    print("\n✅ All models trained successfully!")
    return results
