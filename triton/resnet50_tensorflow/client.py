import numpy as np
import tritonclient.http as httpclient
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tritonclient.utils import triton_to_np_dtype

import time

#def process_image(image_path="dog.jpg"):
def process_image(image_path="header-gulf-birds.jpg"):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)

transformed_img = process_image()

triton_client = httpclient.InferenceServerClient(url="localhost:8000")

inputs = httpclient.InferInput("input_1", transformed_img.shape, datatype="FP32")
inputs.set_data_from_numpy(transformed_img, binary_data=True)

output = httpclient.InferRequestedOutput(
    "predictions", binary_data=True, class_count=1000
)

# Querying the server
a = time.time()
results = triton_client.infer(model_name="resnet50_tensorflow", inputs=[inputs], outputs=[output])
b = time.time()
print("infer time = %f" % (b - a))

predictions = results.as_numpy("predictions").squeeze()

print(predictions[:5])

"""
python client.py 

#def rn50_preprocess(img_path="dog.jpg"):
[b'0.479306:249:MALAMUTE' b'0.204638:248:ESKIMO DOG'
 b'0.083600:250:SIBERIAN HUSKY' b'0.037946:160:AFGHAN HOUND'
 b'0.031664:537:DOGSLED']

#def rn50_preprocess(img_path="header-gulf-birds.jpg"):
infer time = 0.369280
infer time = 0.011506
infer time = 0.011470
infer time = 0.011298
infer time = 0.011375
[b'0.301544:90:LORIKEET' b'0.169441:14:INDIGO FINCH'
 b'0.161432:92:BEE EATER' b'0.093331:94:HUMMINGBIRD'
 b'0.058541:136:EUROPEAN GALLINULE']
perf_analyzer -m resnet50_tensorflow -v -i gRPC -u localhost:8001 -b 1 --max-threads 32 --concurrency-range 2 --shape input_1:1,224,224,3
Concurrency: 2, throughput: 187.49 infer/sec, latency 10663 usec

---config.pbtxt
dynamic_batching {
  preferred_batch_size: [ 2, 4, 8, 16 ]
  max_queue_delay_microseconds: 200000
}
#gpu, tensorrt acc
#def rn50_preprocess(img_path="header-gulf-birds.jpg"):
infer time = 0.471218
infer time = 0.021586
infer time = 0.015530
infer time = 0.014982
infer time = 0.011316
[b'0.301665:90:LORIKEET' b'0.169623:14:INDIGO FINCH'
 b'0.161097:92:BEE EATER' b'0.093255:94:HUMMINGBIRD'
 b'0.058611:136:EUROPEAN GALLINULE']
Concurrency: 2, throughput: 164.377 infer/sec, latency 12161 usec

"""