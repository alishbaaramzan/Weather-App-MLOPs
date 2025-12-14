"""
Script to test the FastAPI endpoints
Run this after starting the API server
"""

import requests
import json

# Base URL
BASE_URL = "http://localhost:8000"


def test_root():
    """Test root endpoint"""
    print("\n" + "="*70)
    print("Testing Root Endpoint: GET /")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_health():
    """Test health endpoint"""
    print("\n" + "="*70)
    print("Testing Health Endpoint: GET /health")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_model_info():
    """Test model info endpoint"""
    print("\n" + "="*70)
    print("Testing Model Info Endpoint: GET /model-info")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/model-info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_predict():
    """Test single prediction endpoint"""
    print("\n" + "="*70)
    print("Testing Prediction Endpoint: POST /predict")
    print("="*70)
    
    # Sample input
    input_data = {
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
    
    print(f"Input: {json.dumps(input_data, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/predict", json=input_data)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_batch_predict():
    """Test batch prediction endpoint"""
    print("\n" + "="*70)
    print("Testing Batch Prediction Endpoint: POST /predict/batch")
    print("="*70)
    
    # Sample batch input
    batch_input = {
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
    
    print(f"Input: {len(batch_input['inputs'])} samples")
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=batch_input)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def run_all_tests():
    """Run all tests"""
    try:
        test_root()
        test_health()
        test_model_info()
        test_predict()
        test_batch_predict()
        
        print("\n" + "="*70)
        print("✅ All tests completed successfully!")
        print("="*70)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API server")
        print("Make sure the API is running: python -m uvicorn app.main:app --reload")
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")


if __name__ == "__main__":
    print("="*70)
    print("Weather Prediction API Test Suite")
    print("="*70)
    print(f"Testing API at: {BASE_URL}")
    print("Make sure the API server is running!")
    print()
    
    run_all_tests()