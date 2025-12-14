"""
API endpoint tests using FastAPI TestClient
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestRootEndpoints:
    """Test root and health endpoints"""
    
    def test_root_endpoint(self):
        """Test GET / returns welcome message"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["version"] == "1.0.0"
    
    def test_health_endpoint(self):
        """Test GET /health returns healthy status"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] in ["healthy", "unhealthy"]
        assert "timestamp" in data
        assert "model_loaded" in data


class TestModelInfoEndpoint:
    """Test model information endpoint"""
    
    def test_model_info_endpoint(self):
        """Test GET /model-info returns model metadata"""
        response = client.get("/model-info")
        assert response.status_code == 200
        
        data = response.json()
        assert "valid_precipitation_classes" in data
        assert "numeric_features" in data
        assert "categorical_features" in data
        assert "model_types" in data
        
        # Check that classes is a list
        assert isinstance(data["valid_precipitation_classes"], list)
        assert len(data["valid_precipitation_classes"]) > 0


class TestPredictionEndpoint:
    """Test prediction endpoints"""
    
    @pytest.fixture
    def valid_input(self):
        """Sample valid input data"""
        return {
            "hour": 9,
            "day_of_week": 1,
            "month": 12,
            "season": 1,
            "Humidity": 0.82,
            "Wind Speed (km/h)": 22.3,
            "Wind Bearing (degrees)": 245,
            "Visibility (km)": 8.5,
            "Pressure (millibars)": 1005.6,
            "Apparent Temperature (C)": -2.3,
            "Summary": "Overcast"
        }
    
    def test_predict_endpoint_valid_input(self, valid_input):
        """Test POST /predict with valid input"""
        response = client.post("/predict", json=valid_input)
        assert response.status_code == 200
        
        data = response.json()
        assert "temperature_celsius" in data
        assert "precipitation_type" in data
        assert "precipitation_probabilities" in data
        assert "confidence" in data
        
        # Validate data types
        assert isinstance(data["temperature_celsius"], (int, float))
        assert isinstance(data["precipitation_type"], str)
        assert isinstance(data["confidence"], float)
        
        # Validate ranges
        assert 0 <= data["confidence"] <= 1
        assert -50 <= data["temperature_celsius"] <= 50  # Reasonable temperature range
    
    def test_predict_endpoint_missing_field(self):
        """Test POST /predict with missing required field"""
        invalid_input = {
            "hour": 9,
            "day_of_week": 1,
            # Missing other required fields
        }
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_invalid_hour(self, valid_input):
        """Test POST /predict with invalid hour value"""
        invalid_input = valid_input.copy()
        invalid_input["hour"] = 25  # Invalid hour
        
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_invalid_month(self, valid_input):
        """Test POST /predict with invalid month value"""
        invalid_input = valid_input.copy()
        invalid_input["month"] = 13  # Invalid month
        
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422
    
    def test_predict_endpoint_invalid_humidity(self, valid_input):
        """Test POST /predict with invalid humidity value"""
        invalid_input = valid_input.copy()
        invalid_input["Humidity"] = 1.5  # Humidity > 1
        
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422


class TestBatchPredictionEndpoint:
    """Test batch prediction endpoint"""
    
    @pytest.fixture
    def valid_batch_input(self):
        """Sample valid batch input"""
        return {
            "inputs": [
                {
                    "hour": 9,
                    "day_of_week": 1,
                    "month": 12,
                    "season": 1,
                    "Humidity": 0.82,
                    "Wind Speed (km/h)": 22.3,
                    "Wind Bearing (degrees)": 245,
                    "Visibility (km)": 8.5,
                    "Pressure (millibars)": 1005.6,
                    "Apparent Temperature (C)": -2.3,
                    "Summary": "Overcast"
                },
                {
                    "hour": 14,
                    "day_of_week": 3,
                    "month": 6,
                    "season": 3,
                    "Humidity": 0.65,
                    "Wind Speed (km/h)": 15.0,
                    "Wind Bearing (degrees)": 180,
                    "Visibility (km)": 10.0,
                    "Pressure (millibars)": 1013.0,
                    "Apparent Temperature (C)": 25.0,
                    "Summary": "Clear"
                }
            ]
        }
    
    def test_batch_predict_endpoint(self, valid_batch_input):
        """Test POST /predict/batch with valid batch input"""
        response = client.post("/predict/batch", json=valid_batch_input)
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert "count" in data
        assert data["count"] == 2
        assert len(data["predictions"]) == 2
        
        # Check each prediction
        for prediction in data["predictions"]:
            assert "temperature_celsius" in prediction
            assert "precipitation_type" in prediction
    
    def test_batch_predict_empty_list(self):
        """Test POST /predict/batch with empty input list"""
        empty_input = {"inputs": []}
        response = client.post("/predict/batch", json=empty_input)
        
        # Should return 200 with empty predictions
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0


class TestErrorHandling:
    """Test error handling"""
    
    def test_404_endpoint(self):
        """Test accessing non-existent endpoint"""
        response = client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_method_not_allowed(self):
        """Test using wrong HTTP method"""
        response = client.get("/predict")  # Should be POST
        assert response.status_code == 405  # Method not allowed


# Performance test (marked as slow)
@pytest.mark.slow
def test_prediction_performance():
    """Test that predictions complete within reasonable time"""
    import time
    
    valid_input = {
        "hour": 12,
        "day_of_week": 2,
        "month": 6,
        "season": 3,
        "Humidity": 0.7,
        "Wind Speed (km/h)": 15.0,
        "Wind Bearing (degrees)": 180,
        "Visibility (km)": 10.0,
        "Pressure (millibars)": 1013.0,
        "Apparent Temperature (C)": 20.0,
        "Summary": "Clear"
    }
    
    start = time.time()
    response = client.post("/predict", json=valid_input)
    duration = time.time() - start
    
    assert response.status_code == 200
    assert duration < 1.0  # Should complete in less than 1 second