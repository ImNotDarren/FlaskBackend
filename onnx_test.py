import numpy as np
import onnxruntime as rt

onnx_session = rt.InferenceSession('PPG_model_4.onnx')
data = np.loadtxt('./test2.csv', delimiter=',', dtype='float32')

input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

print(input_name)

for d in data:
    input_ = np.array([[d.tolist()]]).astype(np.float32)
    PPG_out = onnx_session.run([output_name], {input_name: input_})
    PPG_predicted = PPG_out.argmax(1)
    print(PPG_predicted)
