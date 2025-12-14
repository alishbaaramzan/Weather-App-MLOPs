"""
Utility functions for model saving, loading, and logging
"""

import joblib
import json
import logging
from datetime import datetime
from typing import Dict, Any
import os

logger = logging.getLogger(__name__)


def save_model_artifacts(artifacts: Dict[str, Any],
                        filepath: str = "models/weather_best_models.pkl"):
    """
    Save all model artifacts to a pickle file
    
    Parameters:
    -----------
    artifacts : dict
        Dictionary containing models and metadata
    filepath : str
        Path to save the pickle file
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save artifacts
        joblib.dump(artifacts, filepath)
        logger.info(f"Model artifacts saved to: {filepath}")
        
        # Log artifact contents
        logger.info("Saved artifacts:")
        for key in artifacts.keys():
            logger.info(f"  - {key}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving model artifacts: {str(e)}")
        return False


def load_model_artifacts(filepath: str = "models/weather_best_models.pkl") -> Dict:
    """
    Load model artifacts from pickle file
    
    Parameters:
    -----------
    filepath : str
        Path to the pickle file
        
    Returns:
    --------
    dict
        Loaded artifacts
    """
    try:
        artifacts = joblib.load(filepath)
        logger.info(f"Model artifacts loaded from: {filepath}")
        
        logger.info("Loaded artifacts:")
        for key in artifacts.keys():
            logger.info(f"  - {key}")
        
        return artifacts
        
    except FileNotFoundError:
        logger.error(f"Model file not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading model artifacts: {str(e)}")
        raise


def save_evaluation_results(results: Dict,
                           filepath: str = "models/evaluation_results.json"):
    """
    Save evaluation results to JSON file
    
    Parameters:
    -----------
    results : dict
        Evaluation results
    filepath : str
        Path to save JSON file
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert numpy types to native Python types
        def convert_types(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_types(results)
        
        # Add timestamp
        results_serializable['timestamp'] = datetime.now().isoformat()
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        logger.info(f"Evaluation results saved to: {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving evaluation results: {str(e)}")
        return False


def create_model_artifact_dict(best_classifier: Any,
                               best_regressor: Any,
                               preprocessor: Any,
                               valid_classes: list,
                               numeric_cols: list,
                               categorical_cols: list,
                               evaluation_results: Dict,
                               config: Dict = None) -> Dict:
    """
    Create a complete artifact dictionary for deployment
    
    Parameters:
    -----------
    best_classifier : sklearn model
        Best classification model
    best_regressor : sklearn model
        Best regression model
    preprocessor : ColumnTransformer
        Fitted preprocessor
    valid_classes : list
        Valid precipitation classes
    numeric_cols : list
        Numeric feature names
    categorical_cols : list
        Categorical feature names
    evaluation_results : dict
        Model evaluation metrics
    config : dict
        Additional configuration
        
    Returns:
    --------
    dict
        Complete artifact dictionary
    """
    artifacts = {
        # Models
        'best_classifier': best_classifier,
        'best_regressor': best_regressor,
        'preprocessor': preprocessor,
        
        # Metadata
        'valid_classes': valid_classes,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        
        # Evaluation metrics
        'classifier_metrics': evaluation_results['classifier']['metrics'],
        'regressor_metrics': evaluation_results['regressor']['metrics'],
        
        # Timestamp
        'training_timestamp': datetime.now().isoformat(),
        
        # Version
        'model_version': '1.0.0'
    }
    
    # Add config if provided
    if config:
        artifacts['config'] = config
    
    logger.info("Model artifact dictionary created")
    
    return artifacts


def setup_logging(log_file: str = "logs/training.log",
                 level: int = logging.INFO):
    """
    Setup logging configuration
    
    Parameters:
    -----------
    log_file : str
        Path to log file
    level : int
        Logging level
    """
    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Create formatters and handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logger.info("Logging configured")
    logger.info(f"Log file: {log_file}")


def log_data_summary(df, name: str = "Dataset"):
    """
    Log summary statistics of dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe
    name : str
        Dataset name
    """
    logger.info(f"\n{name} Summary:")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Columns: {list(df.columns)}")
    logger.info(f"  Missing values: {df.isnull().sum().sum()}")
    logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")