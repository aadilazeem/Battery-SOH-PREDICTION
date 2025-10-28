# ab_testing.py - Test new model versions

class ABTest:
    def __init__(self, model_a, model_b, test_duration_days=14):
        self.model_a = model_a
        self.model_b = model_b
        self.test_duration = test_duration_days
        self.results = {
            'model_a': [],
            'model_b': []
        }
    
    def route_prediction(self, features):
        """Route to model A or B (50/50 split)"""
        if np.random.random() < 0.5:
            return 'model_a', self.model_a.predict(features)
        else:
            return 'model_b', self.model_b.predict(features)
    
    def evaluate(self, actual_soh, model_used, prediction):
        """Record prediction result"""
        error = abs(actual_soh - prediction)
        self.results[model_used].append(error)
    
    def get_winner(self):
        """Determine which model performs better"""
        mae_a = np.mean(self.results['model_a'])
        mae_b = np.mean(self.results['model_b'])
        
        improvement = (mae_a - mae_b) / mae_a * 100
        
        if improvement > 2:  # 2% improvement threshold
            return {
                'winner': 'model_b',
                'improvement': improvement,
                'mae_a': mae_a,
                'mae_b': mae_b
            }
        elif improvement < -2:
            return {
                'winner': 'model_a',
                'improvement': abs(improvement),
                'mae_a': mae_a,
                'mae_b': mae_b
            }
        else:
            return {
                'winner': 'tie',
                'improvement': abs(improvement),
                'mae_a': mae_a,
                'mae_b': mae_b
            }

# Usage
test = ABTest(current_model, new_candidate_model)

for battery_data in incoming_predictions:
    model_used, prediction = test.route_prediction(battery_data['features'])
    test.evaluate(battery_data['actual_soh'], model_used, prediction)

results = test.get_winner()
print(f"Winner: {results['winner']} ({results['improvement']:.2f}% improvement)")

if results['winner'] == 'model_b':
    deploy_model_b()