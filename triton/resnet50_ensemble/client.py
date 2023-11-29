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

    results = triton_client.infer('resnet50_ensemble', inputs=inputs, outputs=outputs)
    output_data0 = results.as_numpy('ENSEMBLE_OUTPUT_0')
    print(output_data0.shape)
    print(output_data0)
"""
python client.py

(1, 5)
[['11.534988:250:SIBERIAN HUSKY' '10.306353:248:ESKIMO DOG'
  '10.085402:249:MALAMUTE' '8.162850:174:NORWEGIAN ELKHOUND'
  '7.927982:198:STANDARD SCHNAUZER']]
"""