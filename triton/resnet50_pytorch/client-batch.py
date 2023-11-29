import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from tritonclient.utils import triton_to_np_dtype

import time

if __name__ == '__main__':
    triton_client = httpclient.InferenceServerClient(url='localhost:8000')

    image = Image.open('dog.jpg')

    image = image.resize((224, 224), Image.ANTIALIAS)
    image = np.asarray(image)
    image = image / 255
    image = np.expand_dims(image, axis=0)
    image = np.transpose(image, axes=[0, 3, 1, 2])
    image = image.astype(np.float32)

    inputs = []
    inputs.append(httpclient.InferInput('INPUT__0', image.shape, "FP32"))
    inputs[0].set_data_from_numpy(image, binary_data=False)
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('OUTPUT__0', binary_data=False, class_count=5))

    results = triton_client.infer('resnet50_pytorch', inputs=inputs, outputs=outputs)
    output_data0 = results.as_numpy('OUTPUT__0')
    print(output_data0.shape)
    print(output_data0)
"""
python client-batch.py

(1, 5)
[['11.258604:250:SIBERIAN HUSKY' '10.019813:249:MALAMUTE'
  '9.844797:248:ESKIMO DOG' '8.254933:198:STANDARD SCHNAUZER'
  '8.102138:196:MINIATURE SCHNAUZER']]
"""