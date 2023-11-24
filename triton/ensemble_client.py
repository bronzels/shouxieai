"""
mkdir m1and2_ensemble/1
"""
import time

import numpy as np
import requests
import tritonclient.http as httpclient
from PIL import Image
from tritonclient.utils import np_to_triton_dtype

def main():
    client = httpclient.InferenceServerClient(url="localhost:8000")

    prompts = ["This is a string"]
    text_obj = np.array([prompts], dtype="object")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = np.asarray(Image.open(requests.get(url, stream=True).raw)).astype(
        np.float32
    )
    uint8_array = np.expand_dims(np.array([1, 2, 3], dtype=np.uint8), axis=0)
    int8_array = np.expand_dims(np.array([-1, 2, -3], dtype=np.int8), axis=0)
    image = np.expand_dims(image, axis=0)
    boolean = np.expand_dims(np.array([True]), axis=0)

    input_tensors = [
        httpclient.InferInput(
            "ensemble_input_string", text_obj.shape, np_to_triton_dtype(text_obj.dtype)
        ),
        httpclient.InferInput(
            "ensemble_input_UINT8_array", uint8_array.shape, datatype="UINT8"
        ),
        httpclient.InferInput(
            "ensemble_input_INT8_array", int8_array.shape, datatype="INT8"
        ),
        httpclient.InferInput(
            "ensemble_input_FP32_image", image.shape, datatype="FP32"
        ),
        httpclient.InferInput("ensemble_input_bool", boolean.shape, datatype="BOOL"),
    ]
    input_tensors[0].set_data_from_numpy(text_obj)
    input_tensors[1].set_data_from_numpy(uint8_array)
    input_tensors[2].set_data_from_numpy(int8_array)
    input_tensors[3].set_data_from_numpy(image)
    input_tensors[4].set_data_from_numpy(boolean)

    output = [
        httpclient.InferRequestedOutput("ensemble_output_string"),
        httpclient.InferRequestedOutput("ensemble_output_UINT8_array"),
        httpclient.InferRequestedOutput("ensemble_output_INT8_array"),
        httpclient.InferRequestedOutput("ensemble_output_FP32_image"),
        httpclient.InferRequestedOutput("ensemble_output_bool"),
    ]

    query_response = client.infer(
        model_name="m1and2_ensemble", inputs=input_tensors, outputs=output
    )

    print(query_response.as_numpy("ensemble_output_string"))
    print(query_response.as_numpy("ensemble_output_UINT8_array"))
    print(query_response.as_numpy("ensemble_output_INT8_array"))
    print(query_response.as_numpy("ensemble_output_FP32_image"))
    print(query_response.as_numpy("ensemble_output_bool"))

if __name__ == "__main__":
    main()

"""
python ensemble_client.py
[[b'This is a string']]
[[1 2 3]]
[[-1  2 -3]]
[[[[140.  25.  56.]
   [144.  25.  67.]
   [146.  24.  73.]
   ...
   [ 94.  16.  38.]
   [107.  13.  39.]
   [102.  10.  33.]]

  [[138.  22.  57.]
   [142.  26.  49.]
   [139.  20.  48.]
   ...
   [103.  11.  36.]
   [115.  17.  42.]
   [ 96.  13.  31.]]

  [[135.  22.  42.]
   [150.  33.  59.]
   [142.  23.  53.]
   ...
   [103.   8.  32.]
   [108.  19.  39.]
   [ 93.  10.  26.]]

  ...

  [[237. 100. 190.]
   [225.  84. 196.]
   [236.  96. 203.]
   ...
   [171.  47. 131.]
   [181.  62. 144.]
   [147.  28. 110.]]

  [[230.  84. 221.]
   [226.  80. 213.]
   [238.  99. 202.]
   ...
   [114.  24.  62.]
   [103.   5.  46.]
   [ 89.   9.  44.]]

  [[238. 100. 175.]
   [246. 109. 191.]
   [238.  96. 214.]
   ...
   [ 74.  13.  29.]
   [ 74.  25.  44.]
   [ 73.  17.  42.]]]]
[[ True]]

"""