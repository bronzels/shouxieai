"""
python export_onnx.py

trtexec --onnx=resnet50.onnx \
        --saveEngine=1/model.plan \
        --explicitBatch \
        --useCudaGraph

"""
import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from torchvision import transforms
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

inputs = httpclient.InferInput("input", transformed_img.shape, datatype="FP32")
inputs.set_data_from_numpy(transformed_img, binary_data=True)

outputs = httpclient.InferRequestedOutput(
    "output", binary_data=True, class_count=1000
)

# Querying the server
a = time.time()
results = client.infer(model_name="resnet50_tensorrt", inputs=[inputs], outputs=[outputs])
b = time.time()
print("infer time = %f" % (b - a))
inference_output = results.as_numpy("output")
print(inference_output[:5])

"""
python client.py 

#def rn50_preprocess(img_path="dog.jpg"):
infer time = 0.026027
infer time = 0.007675
infer time = 0.008319
infer time = 0.007597
infer time = 0.007527
[b'13.638241:249:MALAMUTE' b'11.640631:248:ESKIMO DOG'
 b'10.978890:250:SIBERIAN HUSKY' b'10.771481:537:DOGSLED'
 b'7.328541:444:BICYCLE-BUILT-FOR-TWO']

#def rn50_preprocess(img_path="header-gulf-birds.jpg"):
infer time = 0.441362
infer time = 0.031217
infer time = 0.010098
infer time = 0.007665
infer time = 0.007915
[b'12.474083:90:LORIKEET' b'11.525603:92:BEE EATER'
 b'9.661272:14:INDIGO FINCH' b'8.406118:136:EUROPEAN GALLINULE'
 b'8.219951:11:GOLDFINCH']
"""