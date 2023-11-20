import numpy as np
import tritonclient.http as httpclient
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tritonclient.utils import triton_to_np_dtype

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
results = triton_client.infer(model_name="resnet50_tensorflow", inputs=[inputs], outputs=[output])

predictions = results.as_numpy("predictions")
print(predictions[:5])

"""
python client.py 

#def rn50_preprocess(img_path="dog.jpg"):
[b'0.479306:249:MALAMUTE' b'0.204638:248:ESKIMO DOG'
 b'0.083600:250:SIBERIAN HUSKY' b'0.037946:160:AFGHAN HOUND'
 b'0.031664:537:DOGSLED']

#def rn50_preprocess(img_path="header-gulf-birds.jpg"):
[b'0.301544:90:LORIKEET' b'0.169441:14:INDIGO FINCH'
 b'0.161432:92:BEE EATER' b'0.093331:94:HUMMINGBIRD'
 b'0.058541:136:EUROPEAN GALLINULE']
"""