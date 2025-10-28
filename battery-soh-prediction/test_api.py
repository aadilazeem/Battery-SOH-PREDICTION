import requests
import json

BASE_URL = "http://localhost:5000"

print("\n" + "="*60)
print("🧪 Testing Battery SOH API")
print("="*60)

# Test 1: Health Check
print("\n✓ Test 1: Health Check")
response = requests.get(f"{BASE_URL}/health")
print(json.dumps(response.json(), indent=2))

# Test 2: Single Prediction
print("\n✓ Test 2: Single Prediction")
data = {
    "energy_proxy_VAs": 45000,
    "duration_s": 3600,
    "discharge_curvature_coeff": 0.25,
    "voltage_start": 3.8,
    "current_mean": 1.8
}

response = requests.post(
    f"{BASE_URL}/predict",
    json=data,
    headers={"Content-Type": "application/json"}
)

print(json.dumps(response.json(), indent=2))

print("\n" + "="*60)
print("✅ API Tests Complete!")
print("="*60)