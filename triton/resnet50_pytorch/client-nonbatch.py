import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from tritonclient.utils import triton_to_np_dtype

import time

#def rn50_preprocess(img_path="header-gulf-birds.jpg"):
def rn50_preprocess(img_path="dog.jpg"):
    img = Image.open(img_path)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return preprocess(img).numpy()

transformed_img = rn50_preprocess()

client = httpclient.InferenceServerClient(url="localhost:8000")

inputs = httpclient.InferInput("input__0", transformed_img.shape, datatype="FP32")
inputs.set_data_from_numpy(transformed_img, binary_data=True)

outputs = httpclient.InferRequestedOutput(
    "output__0", binary_data=True, class_count=1000
)

# Querying the server
a = time.time()
results = client.infer(model_name="resnet50_pytorch", inputs=[inputs], outputs=[outputs])
b = time.time()
print("infer time = %f" % (b - a))
inference_output = results.as_numpy("output__0")
print(inference_output[:5])

"""
python client.py 
infer time = 0.016423
infer time = 0.009704
infer time = 0.009046
infer time = 0.009671
infer time = 0.009671
#def rn50_preprocess(img_path="dog.jpg"):
[b'13.639860:249:MALAMUTE' b'11.641809:248:ESKIMO DOG'
 b'10.979411:250:SIBERIAN HUSKY' b'10.769635:537:DOGSLED'
 b'7.329067:444:BICYCLE-BUILT-FOR-TWO']

#def rn50_preprocess(img_path="header-gulf-birds.jpg"):
infer time = 0.089917
infer time = 3.075020
infer time = 0.009896
infer time = 0.009726
infer time = 0.009763
[b'12.471457:90:LORIKEET' b'11.527175:92:BEE EATER'
 b'9.658646:14:INDIGO FINCH' b'8.408511:136:EUROPEAN GALLINULE'
 b'8.217762:11:GOLDFINCH']
"""