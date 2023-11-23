"""
python export_onnx.py
mkdir 1
mo --input_model model.onnx \
        --output_dir=1/

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
results = client.infer(model_name="resnet50_openvino", inputs=[inputs], outputs=[outputs])
b = time.time()
print("infer time = %f" % (b - a))
inference_output = results.as_numpy("output")
print(inference_output[:5])

"""
python client.py 

#def rn50_preprocess(img_path="dog.jpg"):
infer time = 0.036798
infer time = 0.028980
infer time = 0.028849
infer time = 0.030025
infer time = 0.029002
[b'13.639914:249:MALAMUTE' b'11.641984:248:ESKIMO DOG'
 b'10.980786:250:SIBERIAN HUSKY' b'10.771177:537:DOGSLED'
 b'7.328811:444:BICYCLE-BUILT-FOR-TWO']

#def rn50_preprocess(img_path="header-gulf-birds.jpg"):
infer time = 0.029533
infer time = 0.031004
infer time = 0.029168
infer time = 0.029412
infer time = 0.029462
[b'12.474465:90:LORIKEET' b'11.525705:92:BEE EATER'
 b'9.660505:14:INDIGO FINCH' b'8.406350:136:EUROPEAN GALLINULE'
 b'8.220250:11:GOLDFINCH']

"""