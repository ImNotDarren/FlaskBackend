import numpy as np
import onnxruntime as rt

onnx_session = rt.InferenceSession('PPG_model.onnx')
data = np.loadtxt('./test2.csv', delimiter=',', dtype='float32')

for d in data:
    input_ = np.array([[d.tolist()]]).astype(np.float32)
    PPG_out = onnx_session.run(['output'], {'input': input_})
    PPG_predicted = PPG_out.argmax(1)
    print(PPG_predicted)
