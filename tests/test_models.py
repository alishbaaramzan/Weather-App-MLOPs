"""
Model validation and quality tests
"""

import pytest
import joblib
import pandas as pd
import numpy as np
from app.prediction import WeatherPredictor


class TestModelArtifacts:
    """Test model artifacts are loaded correctly"""
    
    @pytest.fixture
    def predictor(self):
        """Load predictor instance"""
        return WeatherPredictor()
    
    def test_model_file_exists(self):
        """Test that model file exists"""
        import os
        assert os.path.exists('models/weather_best_models.pkl')
    
    def test_artifacts_loaded(self, predictor):
        """Test that all required artifacts are loaded"""
        artifacts = predictor.artifacts
        
        assert artifacts is not None
        assert 'preprocessor' in artifacts
        assert 'best_classifier' in artifacts
        assert 'best_regressor' in artifacts
        assert 'numeric_cols' in artifacts
        assert 'categorical_cols' in artifacts
        assert 'valid_classes' in artifacts
    
    def test_valid_classes(self, predictor):
        """Test that valid precipitation classes are defined"""
        valid_classes = predictor.artifacts['valid_classes']
        
        assert isinstance(valid_classes, list)
        assert len(valid_classes) > 0
        # Common precipitation types
        assert any(cls in ['rain', 'snow', 'sunny'] for cls in valid_classes)
    
    def test_feature_columns(self, predictor):
        """Test that feature columns are defined"""
        numeric_cols = predictor.artifacts['numeric_cols']
        categorical_cols = predictor.artifacts['categorical_cols']
        
        assert isinstance(numeric_cols, list)
        assert isinstance(categorical_cols, list)
        assert len(numeric_cols) > 0


class TestPredictions:
    """Test prediction functionality"""
    
    @pytest.fixture
    def predictor(self):
        """Load predictor instance"""
        return WeatherPredictor()
    
    @pytest.fixture
    def sample_input(self):
        """Sample input data"""
        return {
            'hour': 12,
            'day_of_week': 2,
            'month': 6,
            'season': 3,
            'Humidity': 0.7,
            'Wind Speed (km/h)': 15.0,
            'Wind Bearing (degrees)': 180,
            'Visibility (km)': 10.0,
            'Pressure (millibars)': 1013.0,
            'Apparent Temperature (C)': 20.0,
            'Summary': 'Clear'
        }
    
    def test_single_prediction(self, predictor, sample_input):
        """Test making a single prediction"""
        result = predictor.predict(sample_input)
        
        assert 'temperature_celsius' in result
        assert 'precipitation_type' in result
        assert 'precipitation_probabilities' in result
        assert 'confidence' in result
        
        # Validate types
        assert isinstance(result['temperature_celsius'], (int, float))
        assert isinstance(result['precipitation_type'], str)
        assert isinstance(result['confidence'], float)
    
    def test_prediction_ranges(self, predictor, sample_input):
        """Test that predictions are within reasonable ranges"""
        result = predictor.predict(sample_input)
        
        # Temperature should be reasonable
        assert -50 <= result['temperature_celsius'] <= 50
        
        # Confidence should be between 0 and 1
        assert 0 <= result['confidence'] <= 1
        
        # Probabilities should sum to approximately 1
        probs = result['precipitation_probabilities']
        prob_sum = sum(probs.values())
        assert 0.99 <= prob_sum <= 1.01
    
    def test_batch_predictions(self, predictor, sample_input):
        """Test making batch predictions"""
        inputs = [sample_input] * 5
        results = predictor.predict_batch(inputs)
        
        assert len(results) == 5
        for result in results:
            assert 'temperature_celsius' in result
            assert 'precipitation_type' in result
    
    def test_prediction_consistency(self, predictor, sample_input):
        """Test that same input produces same output"""
        result1 = predictor.predict(sample_input)
        result2 = predictor.predict(sample_input)
        
        assert result1['temperature_celsius'] == result2['temperature_celsius']
        assert result1['precipitation_type'] == result2['precipitation_type']
        assert result1['confidence'] == result2['confidence']
    
    def test_extreme_values(self, predictor):
        """Test predictions with extreme but valid values"""
        extreme_input = {
            'hour': 0,
            'day_of_week': 0,
            'month': 1,
            'season': 1,
            'Humidity': 0.0,
            'Wind Speed (km/h)': 0.0,
            'Wind Bearing (degrees)': 0,
            'Visibility (km)': 0.0,
            'Pressure (millibars)': 950.0,
            'Apparent Temperature (C)': -40.0,
            'Summary': 'Extreme'
        }
        
        # Should not raise an error
        result = predictor.predict(extreme_input)
        assert 'temperature_celsius' in result


class TestModelPerformance:
    """Test model performance metrics"""
    
    @pytest.fixture
    def evaluation_results(self):
        """Load evaluation results if available"""
        import os
        if os.path.exists('models/evaluation_results.json'):
            import json
            with open('models/evaluation_results.json', 'r') as f:
                return json.load(f)
        return None
    
    def test_classifier_accuracy(self, evaluation_results):
        """Test that classifier meets minimum accuracy threshold"""
        if evaluation_results is None:
            pytest.skip("Evaluation results not available")
        
        clf_metrics = evaluation_results.get('classifier', {}).get('metrics', {})
        accuracy = clf_metrics.get('accuracy', 0)
        
        # Minimum acceptable accuracy
        assert accuracy >= 0.6, f"Classifier accuracy {accuracy} is below threshold 0.6"
    
    def test_regressor_r2(self, evaluation_results):
        """Test that regressor meets minimum R² threshold"""
        if evaluation_results is None:
            pytest.skip("Evaluation results not available")
        
        reg_metrics = evaluation_results.get('regressor', {}).get('metrics', {})
        r2 = reg_metrics.get('r2', 0)
        
        # Minimum acceptable R²
        assert r2 >= 0.5, f"Regressor R² {r2} is below threshold 0.5"
    
    def test_classifier_f1_score(self, evaluation_results):
        """Test classifier F1 score"""
        if evaluation_results is None:
            pytest.skip("Evaluation results not available")
        
        clf_metrics = evaluation_results.get('classifier', {}).get('metrics', {})
        f1 = clf_metrics.get('f1_macro', 0)
        
        assert f1 >= 0.5, f"Classifier F1-score {f1} is below threshold 0.5"


class TestModelInfo:
    """Test model information retrieval"""
    
    def test_get_model_info(self):
        """Test getting model information"""
        predictor = WeatherPredictor()
        info = predictor.get_model_info()
        
        assert 'valid_precipitation_classes' in info
        assert 'numeric_features' in info
        assert 'categorical_features' in info
        assert 'model_types' in info
        
        # Check model types are correct
        model_types = info['model_types']
        assert 'classifier' in model_types
        assert 'regressor' in model_types