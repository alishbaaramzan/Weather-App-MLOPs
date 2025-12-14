"""
Model evaluation functions
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def evaluate_classifier(model: Any,
                        X_test: np.ndarray,
                        y_test: np.ndarray,
                        model_name: str = "Classifier") -> Dict:
    """
    Evaluate classification model
    
    Parameters:
    -----------
    model : sklearn model
        Trained classifier
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    model_name : str
        Name of the model
        
    Returns:
    --------
    dict
        Evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0)
    }
    
    logger.info(f"\n{model_name} Evaluation:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  F1 Score (macro): {metrics['f1_macro']:.4f}")
    logger.info(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    logger.info(f"  Recall (macro): {metrics['recall_macro']:.4f}")
    
    return metrics


def evaluate_regressor(model: Any,
                       X_test: np.ndarray,
                       y_test: np.ndarray,
                       model_name: str = "Regressor") -> Dict:
    """
    Evaluate regression model
    
    Parameters:
    -----------
    model : sklearn model
        Trained regressor
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test targets
    model_name : str
        Name of the model
        
    Returns:
    --------
    dict
        Evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    
    logger.info(f"\n{model_name} Evaluation:")
    logger.info(f"  MSE: {metrics['mse']:.4f}")
    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
    logger.info(f"  MAE: {metrics['mae']:.4f}")
    logger.info(f"  R² Score: {metrics['r2']:.4f}")
    
    return metrics


def evaluate_all_classifiers(models: Dict,
                             X_test: np.ndarray,
                             y_test: np.ndarray) -> pd.DataFrame:
    """
    Evaluate multiple classification models
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained classifiers
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
        
    Returns:
    --------
    pd.DataFrame
        Comparison table of all models
    """
    results = {}
    
    for name, model in models.items():
        metrics = evaluate_classifier(model, X_test, y_test, name)
        results[name] = metrics
    
    results_df = pd.DataFrame(results).T
    logger.info(f"\nClassification Results Comparison:\n{results_df}")
    
    return results_df


def evaluate_all_regressors(models: Dict,
                            X_test: np.ndarray,
                            y_test: np.ndarray) -> pd.DataFrame:
    """
    Evaluate multiple regression models
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained regressors
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test targets
        
    Returns:
    --------
    pd.DataFrame
        Comparison table of all models
    """
    results = {}
    
    for name, model in models.items():
        metrics = evaluate_regressor(model, X_test, y_test, name)
        results[name] = metrics
    
    results_df = pd.DataFrame(results).T
    logger.info(f"\nRegression Results Comparison:\n{results_df}")
    
    return results_df


def get_classification_report(model: Any,
                              X_test: np.ndarray,
                              y_test: np.ndarray,
                              target_names: list = None) -> str:
    """
    Get detailed classification report
    
    Parameters:
    -----------
    model : sklearn model
        Trained classifier
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    target_names : list
        Class names
        
    Returns:
    --------
    str
        Classification report
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=target_names)
    
    logger.info(f"\nDetailed Classification Report:\n{report}")
    
    return report


def get_confusion_matrix(model: Any,
                        X_test: np.ndarray,
                        y_test: np.ndarray) -> np.ndarray:
    """
    Get confusion matrix
    
    Parameters:
    -----------
    model : sklearn model
        Trained classifier
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
        
    Returns:
    --------
    np.ndarray
        Confusion matrix
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    return cm


def get_feature_importance(model: Any,
                          feature_names: list,
                          top_n: int = 15) -> pd.DataFrame:
    """
    Get feature importance from tree-based models
    
    Parameters:
    -----------
    model : sklearn model
        Trained model with feature_importances_
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to return
        
    Returns:
    --------
    pd.DataFrame
        Feature importance dataframe
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have feature_importances_ attribute")
        return None
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:][::-1]
    
    importance_df = pd.DataFrame({
        'feature': [feature_names[i] for i in indices],
        'importance': importances[indices]
    })
    
    logger.info(f"\nTop {top_n} Feature Importances:\n{importance_df}")
    
    return importance_df


def evaluate_final_models(trained_models: Dict,
                         preprocessed_data: Dict,
                         valid_classes: list,
                         feature_names: list) -> Dict:
    """
    Complete evaluation of best models
    
    Parameters:
    -----------
    trained_models : dict
        Dictionary with trained models
    preprocessed_data : dict
        Preprocessed test data
    valid_classes : list
        Valid class names
    feature_names : list
        Feature names
        
    Returns:
    --------
    dict
        All evaluation results
    """
    results = {}
    
    # Evaluate best classifier
    logger.info("\n" + "="*50)
    logger.info("BEST CLASSIFIER EVALUATION")
    logger.info("="*50)
    
    clf_metrics = evaluate_classifier(
        trained_models['best_classifier'],
        preprocessed_data['classification']['X_test'],
        preprocessed_data['classification']['y_test'],
        "Best Classifier"
    )
    
    clf_report = get_classification_report(
        trained_models['best_classifier'],
        preprocessed_data['classification']['X_test'],
        preprocessed_data['classification']['y_test'],
        target_names=valid_classes
    )
    
    clf_cm = get_confusion_matrix(
        trained_models['best_classifier'],
        preprocessed_data['classification']['X_test'],
        preprocessed_data['classification']['y_test']
    )
    
    clf_importance = get_feature_importance(
        trained_models['best_classifier'],
        feature_names,
        top_n=15
    )
    
    results['classifier'] = {
        'metrics': clf_metrics,
        'classification_report': clf_report,
        'confusion_matrix': clf_cm,
        'feature_importance': clf_importance
    }
    
    # Evaluate best regressor
    logger.info("\n" + "="*50)
    logger.info("BEST REGRESSOR EVALUATION")
    logger.info("="*50)
    
    reg_metrics = evaluate_regressor(
        trained_models['best_regressor'],
        preprocessed_data['regression']['X_test'],
        preprocessed_data['regression']['y_test'],
        "Best Regressor"
    )
    
    reg_importance = get_feature_importance(
        trained_models['best_regressor'],
        feature_names,
        top_n=15
    )
    
    results['regressor'] = {
        'metrics': reg_metrics,
        'feature_importance': reg_importance
    }
    
    # Evaluate initial models for comparison
    logger.info("\n" + "="*50)
    logger.info("INITIAL MODELS COMPARISON")
    logger.info("="*50)
    
    clf_comparison = evaluate_all_classifiers(
        trained_models['initial_classifiers'],
        preprocessed_data['classification']['X_test'],
        preprocessed_data['classification']['y_test']
    )
    
    reg_comparison = evaluate_all_regressors(
        trained_models['initial_regressors'],
        preprocessed_data['regression']['X_test'],
        preprocessed_data['regression']['y_test']
    )
    
    results['clf_comparison'] = clf_comparison
    results['reg_comparison'] = reg_comparison
    
    return results