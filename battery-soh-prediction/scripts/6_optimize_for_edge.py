# optimize_for_edge.py - Compress model for IoT devices

import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Convert to ONNX format (much smaller, faster)
initial_types = [('float_input', FloatTensorType([None, 5]))]

onnx_model = convert_sklearn(stacking_gb_final, initial_types=initial_types)
onnx.save_model(onnx_model, 'models/battery_soh_model.onnx')

# Model size comparison
import os
sklearn_size = os.path.getsize('models/battery_soh_ensemble_final.pkl') / 1024  # KB
onnx_size = os.path.getsize('models/battery_soh_model.onnx') / 1024  # KB

print(f"✓ scikit-learn model: {sklearn_size:.1f} KB")
print(f"✓ ONNX model: {onnx_size:.1f} KB")
print(f"✓ Size reduction: {(1 - onnx_size/sklearn_size)*100:.1f}%")

# Inference on edge device
import onnxruntime as rt
import numpy as np
sess = rt.InferenceSession('models/battery_soh_model.onnx')
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# Predict
X_test = np.random.randn(1, 5).astype(np.float32)
soh_pred = sess.run([output_name], {input_name: X_test})[0]
print(f"✓ Edge prediction: {soh_pred}")