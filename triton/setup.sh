nerdctl pull nvcr.io/nvidia/tritonserver:22.04-py3
nerdctl pull nvcr.io/nvidia/tritonserver:22.04-py3-sdk

nerdctl run --gpus all --rm -p8000:8000 -p8001:8001 -p8002:8002 --net=host --rm -v ${PWD}:/models nvcr.io/nvidia/tritonserver:22.04-py3 tritonserver --model-repository=/models

wget -c https://s3.jcloud.sjtu.edu.cn/899a892efef34b1b944a19981040f55b-oss01/pytorch-wheels/cu116/torch-1.12.0+cu116-cp38-cp38-linux_x86_64.whl
wget -c https://mirror.sjtu.edu.cn/pytorch-wheels/cu116/torchvision-0.13.0+cu116-cp38-cp38-linux_x86_64.whl
nerdctl run -it --net=host --rm -v ${PWD}:/models nvcr.io/nvidia/tritonserver:22.04-py3-sdk /bin/bash
  pip install torch-1.12.0+cu116-cp38-cp38-linux_x86_64.whl
  pip install torchvision-0.13.0+cu116-cp38-cp38-linux_x86_64.whl
  pip install tensorflow==2.8.0
  pip install tensorflow-probability==0.16.0
nerdctl commit tritonserver-91d30 tritonserver:22.04-py3-sdk-tchtf

nerdctl run -it --net=host --rm -v ${PWD}:/models tritonserver:22.04-py3-sdk-tchtf /bin/bash
