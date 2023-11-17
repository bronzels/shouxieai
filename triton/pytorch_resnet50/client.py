import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from torchvision import transforms
from tritonclient.utils import triton_to_np_dtype

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

inputs = httpclient.InferInput("input__0", transformed_img.shape, datatype="FP32")
inputs.set_data_from_numpy(transformed_img, binary_data=True)

outputs = httpclient.InferRequestedOutput(
    "output__0", binary_data=True, class_count=1000
)

# Querying the server
results = client.infer(model_name="pytorch_resnet50", inputs=[inputs], outputs=[outputs])
inference_output = results.as_numpy("output__0")
print(inference_output[:5])

"""
python client.py 

#def rn50_preprocess(img_path="dog.jpg"):
[b'13.639860:249:MALAMUTE' b'11.641809:248:ESKIMO DOG'
 b'10.979411:250:SIBERIAN HUSKY' b'10.769635:537:DOGSLED'
 b'7.329067:444:BICYCLE-BUILT-FOR-TWO']

#def rn50_preprocess(img_path="header-gulf-birds.jpg"):
[b'12.471457:90:LORIKEET' b'11.527175:92:BEE EATER'
 b'9.658646:14:INDIGO FINCH' b'8.408511:136:EUROPEAN GALLINULE'
 b'8.217762:11:GOLDFINCH']
"""