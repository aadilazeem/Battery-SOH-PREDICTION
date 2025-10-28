"""
FIXED: REST API for Battery SOH Predictions
Location: scripts/2_prediction_api.py
Purpose: Flask API for model predictions - CORRECTED VERSION
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
from datetime import datetime
import traceback

app = Flask(__name__)

# Global variables to store loaded models
model = None
scaler = None
features = None

def load_models():
    """Load all required model files"""
    global model, scaler, features
    
    print("\n" + "="*80)
    print("LOADING MODEL FILES")
    print("="*80)
    
    try:
        # Load ensemble model
        model_path = 'models/battery_soh_ensemble_final.pkl'
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"✓ Model loaded: {model_path}")
        else:
            print(f"✗ Model not found: {model_path}")
            return False
        
        # Load scaler
        scaler_path = 'models/robust_scaler.pkl'
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(f"✓ Scaler loaded: {scaler_path}")
        else:
            print(f"✗ Scaler not found: {scaler_path}")
            return False
        
        # Load features
        features_path = 'models/top_features.npy'
        if os.path.exists(features_path):
            features_array = np.load(features_path)
            features = list(features_array)
            print(f"✓ Features loaded: {features}")
        else:
            print(f"✗ Features not found: {features_path}")
            return False
        
        print("\n✅ ALL MODELS LOADED SUCCESSFULLY")
        print("="*80 + "\n")
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR LOADING MODELS:")
        print(f"  {str(e)}")
        print(f"\nTraceback:")
        traceback.print_exc()
        print("\n⚠️  SOLUTION: Run 'python scripts/1_save_model.py' first")
        print("="*80 + "\n")
        return False

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    models_ready = all([model is not None, scaler is not None, features is not None])
    
    return jsonify({
        'status': 'OK' if models_ready else 'MODELS_NOT_LOADED',
        'model': 'battery_soh_ensemble_v1.0',
        'models_loaded': models_ready,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    POST endpoint for SOH prediction
    
    Example request:
    {
        "energy_proxy_VAs": 45000,
        "duration_s": 3600,
        "discharge_curvature_coeff": 0.25,
        "voltage_start": 3.8,
        "current_mean": 1.8
    }
    """
    try:
        # Check if models are loaded
        if model is None or scaler is None or features is None:
            return jsonify({
                'error': 'Models not loaded. Run: python scripts/1_save_model.py',
                'model_status': {
                    'model_loaded': model is not None,
                    'scaler_loaded': scaler is not None,
                    'features_loaded': features is not None
                }
            }), 500
        
        data = request.json
        
        if data is None:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate input - check all required features present
        required_features = features
        missing = [f for f in required_features if f not in data]
        
        if missing:
            return jsonify({
                'error': f'Missing features: {missing}',
                'required': required_features,
                'received': list(data.keys())
            }), 400
        
        # Prepare input array in correct order
        X = np.array([[data[f] for f in required_features]])
        
        # Make prediction
        soh_pred = float(model.predict(X)[0])
        
        # Calculate confidence interval (bootstrap)
        predictions = []
        for _ in range(100):
            noise = np.random.normal(0, 0.02, X.shape)
            try:
                pred = float(model.predict(X + noise)[0])
                predictions.append(np.clip(pred, 0.5, 1.0))
            except:
                pass
        
        if predictions:
            ci_lower = float(np.percentile(predictions, 2.5))
            ci_upper = float(np.percentile(predictions, 97.5))
        else:
            ci_lower = float(soh_pred - 0.05)
            ci_upper = float(soh_pred + 0.05)
        
        # Determine health status
        if soh_pred >= 0.90:
            status = "HEALTHY"
            emoji = "🟢"
        elif soh_pred >= 0.80:
            status = "GOOD"
            emoji = "🟡"
        elif soh_pred >= 0.70:
            status = "CAUTION"
            emoji = "🟠"
        else:
            status = "CRITICAL"
            emoji = "🔴"
        
        return jsonify({
            'soh_prediction': soh_pred,
            'confidence_interval': {
                'lower': ci_lower,
                'upper': ci_upper
            },
            'status': f"{emoji} {status}",
            'status_name': status,
            'confidence_width': float(ci_upper - ci_lower),
            'input_features': data,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        print(f"ERROR in predict(): {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch predictions"""
    try:
        if model is None or scaler is None or features is None:
            return jsonify({'error': 'Models not loaded'}), 500
        
        data_list = request.json
        
        if not isinstance(data_list, list):
            return jsonify({'error': 'Expected array of predictions'}), 400
        
        results = []
        for i, data in enumerate(data_list):
            try:
                X = np.array([[data[f] for f in features]])
                soh_pred = float(model.predict(X)[0])
                results.append({
                    'index': i,
                    'soh_prediction': soh_pred,
                    'status': 'OK'
                })
            except Exception as e:
                results.append({
                    'index': i,
                    'error': str(e),
                    'status': 'ERROR'
                })
        
        return jsonify({'predictions': results, 'total': len(results)}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model_name': 'battery_soh_ensemble_v1.0',
        'model_type': 'Stacking Regressor',
        'base_learners': ['RandomForest', 'GradientBoosting', 'XGBoost', 'SVR'],
        'meta_learner': 'GradientBoosting',
        'required_features': features,
        'performance': {
            'r2_score': 0.843,
            'mae': 0.0341,
            'rmse': 0.0381
        },
        'models_loaded': {
            'model': model is not None,
            'scaler': scaler is not None,
            'features': features is not None
        }
    })

if __name__ == '__main__':
    print("\n" + "="*80)
    print("BATTERY SOH PREDICTION API - STARTING UP")
    print("="*80)
    
    # Load models before starting server
    models_loaded = load_models()
    
    if not models_loaded:
        print("\n⚠️  WARNING: Some models failed to load")
        print("The API will still start, but predictions will fail")
        print("Fix: Run 'python scripts/1_save_model.py'\n")
    
    print("\n" + "="*80)
    print("🚀 STARTING FLASK SERVER")
    print("="*80)
    print("\n📍 API Server: http://localhost:5000")
    print("\n🔗 Available Endpoints:")
    print("   GET  /health              → Check API status")
    print("   GET  /model/info          → Model information")
    print("   POST /predict             → Single prediction")
    print("   POST /predict/batch       → Batch predictions")
    print("\n📝 Example Request:")
    print("""
    POST /predict
    Content-Type: application/json
    
    {
      "energy_proxy_VAs": 45000,
      "duration_s": 3600,
      "discharge_curvature_coeff": 0.25,
      "voltage_start": 3.8,
      "current_mean": 1.8
    }
    """)
    print("="*80)
    print("Press CTRL+C to stop\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)