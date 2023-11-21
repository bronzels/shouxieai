"""
mkdir -p 1
wget -O 1/model.onnx \
     https://contentmamluswest001.blob.core.windows.net/content/14b2744cf8d6418c87ffddc3f3127242/9502630827244d60a1214f250e3bbca7/08aed7327d694b8dbaee2c97b8d0fcba/densenet121-1.2.onnx
"""

import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from torchvision import transforms
from tritonclient.utils import triton_to_np_dtype

import time

#def rn50_preprocess(img_path="dog.jpg"):
def rn50_preprocess(img_path="header-gulf-birds.jpg"):
    img = Image.open(img_path)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return np.expand_dims(preprocess(img).numpy(), axis=0)

transformed_img = rn50_preprocess()

client = httpclient.InferenceServerClient(url="localhost:8000")

inputs = httpclient.InferInput("data_0", transformed_img.shape, datatype="FP32")
inputs.set_data_from_numpy(transformed_img, binary_data=True)

outputs = httpclient.InferRequestedOutput("fc6_1", binary_data=True, class_count=1000)

# Querying the server
a = time.time()
results = client.infer(model_name="densenet_onnx", inputs=[inputs], outputs=[outputs])
b = time.time()
print("infer time = %f" % (b - a))
inference_output = results.as_numpy("fc6_1").astype(str)

print(np.squeeze(inference_output)[:5])

"""
python client.py 
infer time = 0.260158
infer time = 0.025959
infer time = 0.011724
infer time = 0.011970
infer time = 0.012032
#def rn50_preprocess(img_path="dog.jpg"):
['8.687009:250:SIBERIAN HUSKY' '8.120752:249:MALAMUTE'
 '7.818727:248:ESKIMO DOG' '7.177729:169:BORZOI' '6.894028:172:WHIPPET']

#def rn50_preprocess(img_path="header-gulf-birds.jpg"):
infer time = 0.022623
infer time = 0.012144
infer time = 0.012181
infer time = 0.012062
infer time = 0.012135
['11.548535:92:BEE EATER' '11.231399:14:INDIGO FINCH'
 '7.529447:95:JACAMAR' '6.922517:17:JAY' '6.575730:88:MACAW']
"""
