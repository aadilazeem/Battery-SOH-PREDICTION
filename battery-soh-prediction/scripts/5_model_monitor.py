# model_monitor.py - Track model degradation

class ModelMonitor:
    def __init__(self, baseline_mae=0.0341, drift_threshold=0.05):
        self.baseline_mae = baseline_mae
        self.drift_threshold = drift_threshold
        self.prediction_history = []
        
    def log_prediction(self, actual_soh, predicted_soh, timestamp=None):
        """Log prediction for monitoring"""
        error = abs(actual_soh - predicted_soh)
        
        self.prediction_history.append({
            'timestamp': timestamp or datetime.now(),
            'actual': actual_soh,
            'predicted': predicted_soh,
            'error': error,
            'relative_error': error / actual_soh if actual_soh > 0 else 0
        })
    
    def check_drift(self, window_size=30):
        """
        Detect model drift
        Alert if recent MAE > baseline * (1 + drift_threshold)
        """
        if len(self.prediction_history) < window_size:
            return {'drift_detected': False, 'reason': 'Insufficient data'}
        
        recent = self.prediction_history[-window_size:]
        recent_mae = np.mean([p['error'] for p in recent])
        
        threshold = self.baseline_mae * (1 + self.drift_threshold)
        drift_detected = recent_mae > threshold
        
        return {
            'drift_detected': drift_detected,
            'baseline_mae': self.baseline_mae,
            'recent_mae': recent_mae,
            'threshold': threshold,
            'drift_percentage': ((recent_mae - self.baseline_mae) / self.baseline_mae * 100)
        }
    
    def trigger_retraining(self):
        """
        Trigger model retraining if drift detected
        Return: True if retraining needed, False otherwise
        """
        drift_info = self.check_drift()
        
        if drift_info['drift_detected']:
            print(f"⚠️ DRIFT ALERT: Model MAE increased by {drift_info['drift_percentage']:.1f}%")
            print(f"   Baseline: {drift_info['baseline_mae']:.6f}")
            print(f"   Recent: {drift_info['recent_mae']:.6f}")
            print(f"   Threshold: {drift_info['threshold']:.6f}")
            print("\n🔄 Initiating model retraining...")
            return True
        
        return False
    
    def generate_report(self):
        """Generate monitoring report"""
        errors = np.array([p['error'] for p in self.prediction_history])
        
        return {
            'total_predictions': len(self.prediction_history),
            'mean_error': errors.mean(),
            'std_error': errors.std(),
            'max_error': errors.max(),
            'percentile_95': np.percentile(errors, 95),
            'drift_status': self.check_drift()
        }

# Usage
monitor = ModelMonitor(baseline_mae=0.0341)

# Simulate incoming predictions
for actual, predicted in predictions:
    monitor.log_prediction(actual, predicted)
    
    if monitor.trigger_retraining():
        # Automatically retrain model
        retrain_ensemble_model()

# Generate report
report = monitor.generate_report()
print(report)