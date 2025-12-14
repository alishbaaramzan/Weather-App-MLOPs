"""
Prediction pipeline for weather forecasting
Loads trained models and makes predictions
"""

import joblib
import pandas as pd
import numpy as np
from typing import Dict, Union
import logging

logger = logging.getLogger(__name__)


class WeatherPredictor:
    """Weather prediction class that handles model loading and predictions"""
    
    def __init__(self, model_path: str = 'models/weather_best_models.pkl'):
        """
        Initialize predictor and load model artifacts
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model pickle file
        """
        self.model_path = model_path
        self.artifacts = None
        self.load_models()
    
    def load_models(self):
        """Load all model artifacts from pickle file"""
        try:
            self.artifacts = joblib.load(self.model_path)
            logger.info(f"Model artifacts loaded successfully from {self.model_path}")
            logger.info(f"Valid precipitation classes: {self.artifacts['valid_classes']}")
        except FileNotFoundError:
            logger.error(f"Model file not found: {self.model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def predict(self, input_data: Union[Dict, pd.DataFrame]) -> Dict:
        """
        Make weather predictions
        
        Parameters:
        -----------
        input_data : dict or pd.DataFrame
            Input features. Expected features:
            - hour (int): Hour of day (0-23)
            - day_of_week (int): Day of week (0-6)
            - month (int): Month (1-12)
            - season (int): Season (1-4)
            - Humidity (float): Humidity value
            - Wind Speed (km/h) (float)
            - Wind Bearing (degrees) (float)
            - Visibility (km) (float)
            - Pressure (millibars) (float)
            - Apparent Temperature (C) (float)
            - Summary (str): Weather summary text
            
        Returns:
        --------
        dict
            Predictions with:
            - temperature_celsius: Predicted temperature
            - precipitation_type: Predicted precipitation type
            - precipitation_probabilities: Probability for each class
            - confidence: Confidence of precipitation prediction
        """
        if self.artifacts is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Convert dict to DataFrame
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        # Get required columns
        required_cols = self.artifacts['numeric_cols'] + self.artifacts['categorical_cols']
        
        # Fill missing columns with defaults
        for col in required_cols:
            if col not in input_df.columns:
                # Use 0 for numeric, 'Unknown' for categorical
                if col in self.artifacts['numeric_cols']:
                    input_df[col] = 0
                    logger.warning(f"Missing numeric column '{col}', filled with 0")
                else:
                    input_df[col] = 'Unknown'
                    logger.warning(f"Missing categorical column '{col}', filled with 'Unknown'")
        
        # Select and order columns correctly
        input_df = input_df[required_cols]
        
        # Preprocess the input
        input_processed = self.artifacts['preprocessor'].transform(input_df)
        
        # Make predictions
        temp_prediction = self.artifacts['best_regressor'].predict(input_processed)[0]
        precip_prediction = self.artifacts['best_classifier'].predict(input_processed)[0]
        precip_probabilities = self.artifacts['best_classifier'].predict_proba(input_processed)[0]
        
        # Format results
        result = {
            'temperature_celsius': round(float(temp_prediction), 2),
            'precipitation_type': str(precip_prediction),
            'precipitation_probabilities': {
                str(cls): round(float(prob), 4)
                for cls, prob in zip(self.artifacts['valid_classes'], precip_probabilities)
            },
            'confidence': round(float(max(precip_probabilities)), 4)
        }
        
        return result
    
    def predict_batch(self, input_data_list: list) -> list:
        """
        Make predictions for multiple inputs
        
        Parameters:
        -----------
        input_data_list : list
            List of input dictionaries
            
        Returns:
        --------
        list
            List of prediction dictionaries
        """
        results = []
        for input_data in input_data_list:
            try:
                result = self.predict(input_data)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting for input: {str(e)}")
                results.append({
                    'error': str(e),
                    'input': input_data
                })
        
        return results
    
    def get_model_info(self) -> Dict:
        """
        Get information about loaded models
        
        Returns:
        --------
        dict
            Model metadata
        """
        if self.artifacts is None:
            raise RuntimeError("Models not loaded.")
        
        return {
            'valid_precipitation_classes': self.artifacts['valid_classes'],
            'numeric_features': self.artifacts['numeric_cols'],
            'categorical_features': self.artifacts['categorical_cols'],
            'temperature_column': self.artifacts.get('temp_col', 'Temperature (C)'),
            'precipitation_column': self.artifacts.get('precip_col', 'Precip Type'),
            'model_types': {
                'classifier': str(type(self.artifacts['best_classifier']).__name__),
                'regressor': str(type(self.artifacts['best_regressor']).__name__)
            }
        }


# Singleton instance (loaded once at startup)
_predictor_instance = None


def get_predictor(model_path: str = 'models/weather_best_models.pkl') -> WeatherPredictor:
    """
    Get or create predictor singleton instance
    
    Parameters:
    -----------
    model_path : str
        Path to model file
        
    Returns:
    --------
    WeatherPredictor
        Predictor instance
    """
    global _predictor_instance
    
    if _predictor_instance is None:
        _predictor_instance = WeatherPredictor(model_path)
    
    return _predictor_instance