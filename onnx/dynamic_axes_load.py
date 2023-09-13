import onnxruntime
import numpy as np

origin_tensor = np.random.rand(1, 3, 10, 10).astype(np.float32)
multi_batch_tensor = np.random.rand(2, 3, 10, 10).astype(np.float32)
big_tensor = np.random.rand(1, 3, 20, 20).astype(np.float32)

inputs = [origin_tensor, multi_batch_tensor, big_tensor]
exceptions = dict()

model_names = ['model_static.onnx',
'model_dynamic_0.onnx',
'model_dynamic_23.onnx']

for model_name in model_names:
    for i, input in enumerate(inputs):
        #print('i:', i, ',  input: ', input)
        try:
            ort_session = onnxruntime.InferenceSession(model_name)
            ort_inputs = {'in': input}
            ort_session.run(['out'], ort_inputs)
        except Exception as e:
            exceptions[(i, model_name)] = e
            print(f'Input[{i}] on model {model_name} error.')
        else:
            print(f'Input[{i}] on model {model_name} succeed.')

