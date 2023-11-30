import numpy as np
import tritonclient.http as httpclient
import torch
from PIL import Image

import time

if __name__ == '__main__':
    triton_client = httpclient.InferenceServerClient(url='localhost:8000')

    image = Image.open('dog.jpg')
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    inputs = []
    inputs.append(httpclient.InferInput('ENSEMBLE_INPUT_0', image.shape, "FP32"))
    inputs[0].set_data_from_numpy(image, binary_data=False)
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('ENSEMBLE_OUTPUT_0', binary_data=False, class_count=5))

    a = time.time()
    results = triton_client.infer('resnet50_ensemble', inputs=inputs, outputs=outputs)
    b = time.time()
    print("infer time = %f" % (b - a))
    output_data0 = results.as_numpy('ENSEMBLE_OUTPUT_0')
    print(output_data0.shape)
    print(output_data0)
"""
python client.py

config-pytorch.pbtxt
(1, 5)
[['11.534988:250:SIBERIAN HUSKY' '10.306353:248:ESKIMO DOG'
  '10.085402:249:MALAMUTE' '8.162850:174:NORWEGIAN ELKHOUND'
  '7.927982:198:STANDARD SCHNAUZER']]

dali device='cpu'
config-pythontorch.pbtxt w/n FORCE_CPU_ONLY_INPUT_TENSORS=no
input_tensor.is_cpu():True
infer time = 0.299005
infer time = 0.284388
infer time = 0.291592
infer time = 0.278314
infer time = 0.274090
(1, 5)
[['11.244161:250:SIBERIAN HUSKY' '10.023156:248:ESKIMO DOG'
  '9.782356:249:MALAMUTE' '7.993082:198:STANDARD SCHNAUZER'
  '7.964351:174:NORWEGIAN ELKHOUND']]

dali device='cpu'
config-pythontorch.pbtxt wth FORCE_CPU_ONLY_INPUT_TENSORS=no
tritonserver logs:
input_tensor.is_cpu():True
infer time = 1.631548
infer time = 0.297429
infer time = 0.303935
infer time = 0.273702
infer time = 0.273770
(1, 5)
[['11.244161:250:SIBERIAN HUSKY' '10.023156:248:ESKIMO DOG'
  '9.782356:249:MALAMUTE' '7.993082:198:STANDARD SCHNAUZER'
  '7.964351:174:NORWEGIAN ELKHOUND']]

dali device='gpu'
config-pythontorch.pbtxt w/n FORCE_CPU_ONLY_INPUT_TENSORS=no
tritonserver logs:
input_tensor.is_cpu():True
infer time = 1.808095
infer time = 0.302512
infer time = 0.288057
infer time = 0.265305
infer time = 0.272057
(1, 5)
[['11.243821:250:SIBERIAN HUSKY' '10.023777:248:ESKIMO DOG'
  '9.781121:249:MALAMUTE' '7.991022:198:STANDARD SCHNAUZER'
  '7.964470:174:NORWEGIAN ELKHOUND']]

dali device='gpu'
config-pythontorch.pbtxt wth FORCE_CPU_ONLY_INPUT_TENSORS=no
tritonserver logs:
input_tensor.is_cpu():False
infer time = 1.688249
infer time = 0.318237
infer time = 0.270673
infer time = 0.270673
infer time = 0.271146
(1, 5)
[['11.243821:250:SIBERIAN HUSKY' '10.023777:248:ESKIMO DOG'
  '9.781121:249:MALAMUTE' '7.991022:198:STANDARD SCHNAUZER'
  '7.964470:174:NORWEGIAN ELKHOUND']]


"""