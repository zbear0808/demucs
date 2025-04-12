import onnxscript

import onnx

# run model optimizer

#as of right now it looks like the dependencies are not installed in the env

# so i'm running this in a separate env with just onnx and onnxscript installed
# i don't want to break my demucs env


model = onnx.load("htdemucs_ft.onnx")

print('checking model')
ch = onnx.checker.check_model(model, full_check=True, check_custom_domain=True)
print('model is valid:', ch)

opt = onnxscript.optimizer.optimize(model)

onnx.save(opt, "htdemucs_optimized.onnx")