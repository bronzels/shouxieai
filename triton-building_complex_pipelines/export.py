"""
#fq on mmubu
export http_proxy=http://mmubu:10792;export https_proxy=http://mmubu:10792;
export HTTP_PROXY=${http_proxy};export HTTPS_PROXY=${https_proxy};
#https://huggingface.co/settings/tokens
#set token, copy token to clipboard
git config --global credential.helper store
huggingface-cli login
#input token
#input yes

#start client to export
nerdctl run --runtime=nvidia-container-runtime -it --net=host --rm -v /usr/local/cuda:/usr/local/cuda -v `realpath $PWD/../data`:/data -v ${PWD}:/models.. tritonserver:${triton_version}-py3-sdk-tchtf /bin/bash
"""

import torch
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

prompt = "Draw a dog"
vae = AutoencoderKL.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="vae", use_auth_token=True
)

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

vae.forward = vae.decode
torch.onnx.export(
    vae,
    (torch.randn(1, 4, 64, 64), False),
    "vae.onnx",
    input_names=["latent_sample", "return_dict"],
    output_names=["sample"],
    dynamic_axes={
        "latent_sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
    },
    do_constant_folding=True,
    opset_version=14,
)

text_input = tokenizer(
    prompt,
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)

torch.onnx.export(
    text_encoder,
    (text_input.input_ids.to(torch.int32)),
    "./model_repository/text_encoder/1/model.onnx",
    input_names=["input_ids"],
    output_names=["last_hidden_state", "pooler_output"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
    },
    opset_version=14,
    do_constant_folding=True,
)

"""
pip uninstall urllib3==2.1.0 -y
pip install urllib3==2.1.0
huggingface-cli login
python export.py

requests           2.31.0
pip install requests==2.21.0

trtexec --onnx=vae.onnx --saveEngine=./model_repository/vae/1/model.plan
--minShapes=latent_sample:1x4x64x64
--optShapes=latent_sample:4x4x64x64
--maxShapes=latent_sample:8x4x64x64 --fp16

nerdctl run --gpus all --shm-size=1g --rm -p8000:8000 -p8001:8001 -p8002:8002 --net=host --rm -v ${PWD}/model_repository:/models tritonserver:${triton_version}-py3-tchtf tritonserver --model-repository=/models --model-control-mode poll


"""