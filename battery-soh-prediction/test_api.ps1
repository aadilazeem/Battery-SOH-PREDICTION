# Simple API Test - No special characters

Write-Host "Testing Battery SOH API"
Write-Host "=============================="

# Test 1: Health Check
Write-Host ""
Write-Host "Test 1: Health Check"

$health = Invoke-WebRequest -Uri "http://localhost:5000/health" -Method GET
Write-Host $health.Content

# Test 2: Single Prediction
Write-Host ""
Write-Host "Test 2: Single Prediction"

$data = @{
    energy_proxy_VAs = 45000
    duration_s = 3600
    discharge_curvature_coeff = 0.25
    voltage_start = 3.8
    current_mean = 1.8
} | ConvertTo-Json

$response = Invoke-WebRequest `
    -Uri "http://localhost:5000/predict" `
    -Method POST `
    -Headers @{"Content-Type"="application/json"} `
    -Body $data

Write-Host $response.Content

Write-Host ""
Write-Host "=============================="
Write-Host "Tests Complete!"