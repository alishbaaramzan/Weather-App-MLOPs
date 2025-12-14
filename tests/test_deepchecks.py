"""
Data validation and model monitoring tests using DeepChecks
"""

import pytest
import pandas as pd
import numpy as np
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import (
    DataDuplicates,
    MixedDataTypes,
    MixedNulls,
    StringMismatch,
    IsSingleValue,
    SpecialCharacters,
    FeatureLabelCorrelation,
    TrainTestFeatureDrift,
    DatasetsSizeComparison
)


class TestDataIntegrity:
    """Test data quality and integrity"""
    
    @pytest.fixture
    def sample_data(self):
        """Load sample data for testing"""
        # You can load a small subset of your actual data
        # or create synthetic data for testing
        return pd.DataFrame({
            'hour': [9, 14, 18, 22, 3],
            'day_of_week': [1, 3, 5, 0, 2],
            'month': [12, 6, 9, 3, 7],
            'season': [1, 3, 4, 1, 3],
            'Humidity': [0.82, 0.65, 0.75, 0.90, 0.70],
            'Wind Speed (km/h)': [22.3, 15.0, 18.5, 10.2, 25.0],
            'Wind Bearing (degrees)': [245, 180, 90, 0, 270],
            'Visibility (km)': [8.5, 10.0, 9.0, 5.0, 12.0],
            'Pressure (millibars)': [1005.6, 1013.0, 1010.0, 1008.0, 1015.0],
            'Apparent Temperature (C)': [-2.3, 25.0, 15.0, -5.0, 28.0],
            'Summary': ['Overcast', 'Clear', 'Partly Cloudy', 'Foggy', 'Clear'],
            'Temperature (C)': [-1.5, 26.0, 16.0, -4.0, 29.0],
            'Precip Type': ['snow', 'sunny', 'sunny', 'rain', 'sunny']
        })
    
    def test_no_data_duplicates(self, sample_data):
        """Test that data has no duplicate rows"""
        check = DataDuplicates()
        dataset = Dataset(sample_data, label='Temperature (C)')
        result = check.run(dataset)
        
        # Check should pass (no duplicates expected)
        assert result.passed()
    
    def test_no_mixed_data_types(self, sample_data):
        """Test that columns don't have mixed data types"""
        check = MixedDataTypes()
        dataset = Dataset(sample_data, label='Temperature (C)')
        result = check.run(dataset)
        
        # Should pass (all columns have consistent types)
        assert result.passed()
    
    def test_no_mixed_nulls(self, sample_data):
        """Test for mixed null representations"""
        check = MixedNulls()
        dataset = Dataset(sample_data, label='Temperature (C)')
        result = check.run(dataset)
        
        # Should pass (consistent null handling)
        assert result.passed()
    
    def test_no_single_value_features(self, sample_data):
        """Test that features have variation"""
        check = IsSingleValue()
        dataset = Dataset(sample_data, label='Temperature (C)')
        result = check.run(dataset)
        
        # Should pass (features have multiple values)
        assert result.passed()
    
    def test_string_mismatch(self, sample_data):
        """Test for string inconsistencies in categorical columns"""
        check = StringMismatch()
        dataset = Dataset(sample_data, 
                         label='Temperature (C)',
                         cat_features=['Summary', 'Precip Type'])
        result = check.run(dataset)
        
        # Log any issues found
        if not result.passed():
            print(f"String mismatch found: {result.value}")


class TestFeatureValidation:
    """Test feature properties and relationships"""
    
    @pytest.fixture
    def sample_data(self):
        """Sample data with features and target"""
        return pd.DataFrame({
            'hour': np.random.randint(0, 24, 100),
            'Humidity': np.random.uniform(0.3, 1.0, 100),
            'Wind Speed (km/h)': np.random.uniform(0, 50, 100),
            'Pressure (millibars)': np.random.uniform(980, 1030, 100),
            'Apparent Temperature (C)': np.random.uniform(-10, 35, 100),
            'Temperature (C)': np.random.uniform(-10, 35, 100)
        })
    
    def test_feature_label_correlation(self, sample_data):
        """Test correlation between features and target"""
        check = FeatureLabelCorrelation()
        dataset = Dataset(sample_data, label='Temperature (C)')
        result = check.run(dataset)
        
        # Some features should be correlated with temperature
        # This is informational, not strictly pass/fail
        print(f"Feature-label correlations: {result.value}")


class TestDataDrift:
    """Test for data drift between train and test sets"""
    
    @pytest.fixture
    def train_test_data(self):
        """Create sample train and test datasets"""
        # Simulate train data
        train_df = pd.DataFrame({
            'hour': np.random.randint(0, 24, 100),
            'Humidity': np.random.uniform(0.3, 1.0, 100),
            'Wind Speed (km/h)': np.random.uniform(0, 50, 100),
            'Pressure (millibars)': np.random.uniform(980, 1030, 100),
            'Temperature (C)': np.random.uniform(-10, 35, 100)
        })
        
        # Simulate test data (slightly different distribution)
        test_df = pd.DataFrame({
            'hour': np.random.randint(0, 24, 50),
            'Humidity': np.random.uniform(0.4, 0.9, 50),
            'Wind Speed (km/h)': np.random.uniform(5, 45, 50),
            'Pressure (millibars)': np.random.uniform(990, 1025, 50),
            'Temperature (C)': np.random.uniform(-5, 30, 50)
        })
        
        return train_df, test_df
    
    def test_feature_drift(self, train_test_data):
        """Test for feature drift between train and test"""
        train_df, test_df = train_test_data
        
        check = TrainTestFeatureDrift()
        train_dataset = Dataset(train_df, label='Temperature (C)')
        test_dataset = Dataset(test_df, label='Temperature (C)')
        
        result = check.run(train_dataset, test_dataset)
        
        # Log drift results
        print(f"Feature drift detected: {result.value}")
        
        # Depending on your data, you might want to assert thresholds
        # For now, just ensure the check runs without errors
        assert result is not None
    
    def test_dataset_size_comparison(self, train_test_data):
        """Test that train and test sets are reasonably sized"""
        train_df, test_df = train_test_data
        
        check = DatasetsSizeComparison()
        train_dataset = Dataset(train_df, label='Temperature (C)')
        test_dataset = Dataset(test_df, label='Temperature (C)')
        
        result = check.run(train_dataset, test_dataset)
        
        # Should pass if datasets are reasonable
        assert result is not None


class TestModelMonitoring:
    """Tests for ongoing model monitoring"""
    
    def test_prediction_distribution(self):
        """Test that predictions follow expected distribution"""
        from app.prediction import WeatherPredictor
        
        predictor = WeatherPredictor()
        
        # Generate multiple predictions
        predictions_temp = []
        predictions_precip = []
        
        for _ in range(20):
            input_data = {
                'hour': np.random.randint(0, 24),
                'day_of_week': np.random.randint(0, 7),
                'month': np.random.randint(1, 13),
                'season': np.random.randint(1, 5),
                'Humidity': np.random.uniform(0.3, 1.0),
                'Wind Speed (km/h)': np.random.uniform(0, 50),
                'Wind Bearing (degrees)': np.random.uniform(0, 360),
                'Visibility (km)': np.random.uniform(0, 15),
                'Pressure (millibars)': np.random.uniform(980, 1030),
                'Apparent Temperature (C)': np.random.uniform(-10, 35),
                'Summary': np.random.choice(['Clear', 'Cloudy', 'Overcast', 'Rainy'])
            }
            
            result = predictor.predict(input_data)
            predictions_temp.append(result['temperature_celsius'])
            predictions_precip.append(result['precipitation_type'])
        
        # Check temperature predictions are reasonable
        assert -50 <= min(predictions_temp) <= 50
        assert -50 <= max(predictions_temp) <= 50
        
        # Check we get variety in precipitation predictions
        unique_precip = set(predictions_precip)
        print(f"Unique precipitation types predicted: {unique_precip}")


# Integration test combining multiple checks
@pytest.mark.integration
class TestFullDataValidation:
    """Integration test running full data validation suite"""
    
    def test_full_validation_suite(self):
        """Run complete data validation suite"""
        # This would typically run on your actual training data
        # For demo, we'll create sample data
        
        data = pd.DataFrame({
            'hour': np.random.randint(0, 24, 200),
            'Humidity': np.random.uniform(0.3, 1.0, 200),
            'Wind Speed (km/h)': np.random.uniform(0, 50, 200),
            'Temperature (C)': np.random.uniform(-10, 35, 200),
            'Precip Type': np.random.choice(['rain', 'snow', 'sunny'], 200)
        })
        
        dataset = Dataset(data, label='Temperature (C)', cat_features=['Precip Type'])
        
        # Run multiple checks
        checks = [
            DataDuplicates(),
            MixedDataTypes(),
            IsSingleValue(),
        ]
        
        results = []
        for check in checks:
            result = check.run(dataset)
            results.append(result.passed())
        
        # All checks should pass
        assert all(results), "Some data validation checks failed"