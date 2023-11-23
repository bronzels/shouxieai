#triton_version=22.04
triton_version=22.11
echo "export triton_version=${triton_version}" >> ~/.bashrc
nerdctl pull nvcr.io/nvidia/tritonserver:${triton_version}-py3
nerdctl pull nvcr.io/nvidia/tritonserver:${triton_version}-py3-sdk

nerdctl run --gpus all --rm -p8000:8000 -p8001:8001 -p8002:8002 --net=host --rm -v ${PWD}:/models nvcr.io/nvidia/tritonserver:${triton_version}-py3 tritonserver --model-repository=/models
nerdctl exec -it `nerdctl ps -a | grep tritonserver:${triton_version}-py3 | grep "8000->8000" | awk '{print $1}'` /bin/bash
  cd /models
  pip install torch-1.12.0+cu116-cp38-cp38-linux_x86_64.whl
  pip install torchvision-0.13.0+cu116-cp38-cp38-linux_x86_64.whl
nerdctl commit `nerdctl ps -a | grep tritonserver:${triton_version}-py3 | grep "8000->8000" | awk '{print $1}'` tritonserver:${triton_version}-py3-tchtf
nerdctl run --gpus all --rm -p8000:8000 -p8001:8001 -p8002:8002 --net=host --rm -v ${PWD}:/models tritonserver:${triton_version}-py3-tchtf tritonserver --model-repository=/models

nerdctl exec -it `nerdctl ps -a | grep tritonserver:${triton_version}-py3-tchtf | awk '{print $1}'` /bin/bash

wget -c https://s3.jcloud.sjtu.edu.cn/899a892efef34b1b944a19981040f55b-oss01/pytorch-wheels/cu116/torch-1.12.0+cu116-cp38-cp38-linux_x86_64.whl
wget -c https://mirror.sjtu.edu.cn/pytorch-wheels/cu116/torchvision-0.13.0+cu116-cp38-cp38-linux_x86_64.whl
nerdctl run -it --net=host --rm -v ${PWD}:/models nvcr.io/nvidia/tritonserver:${triton_version}-py3-sdk /bin/bash
  cd /models
  pip install torch-1.12.0+cu116-cp38-cp38-linux_x86_64.whl
  pip install torchvision-0.13.0+cu116-cp38-cp38-linux_x86_64.whl
  pip install tensorflow==2.8.0
  pip install tensorflow-probability==0.16.0
  pip install validators==0.22.0
  pip install openvino-dev==2022.1.0
nerdctl commit `nerdctl ps -a | grep tritonserver:${triton_version}-py3-sdk | awk '{print $1}'` tritonserver:${triton_version}-py3-sdk-tchtf
nerdctl stop `nerdctl ps -a | grep tritonserver:${triton_version}-py3 | awk '{print $1}'`
nerdctl rm `nerdctl ps -a | grep tritonserver:${triton_version}-py3 | awk '{print $1}'`

nerdctl exec -it `nerdctl ps -a | grep tritonserver:${triton_version}-py3-sdk | awk '{print $1}'` /bin/bash

nerdctl run -it --net=host --rm -v `realpath $PWD/../data`:/data -v ${PWD}:/models tritonserver:${triton_version}-py3-sdk-tchtf /bin/bash
  cd /models

nerdctl restart `nerdctl ps -a | grep "tritonserver:${triton_version}-py3-tchtf" | awk '{print $1}'`
nerdctl logs -f `nerdctl ps -a | grep "tritonserver:${triton_version}-py3-tchtf" | awk '{print $1}'`

nerdctl start `nerdctl ps -a | grep "tritonserver:${triton_version}-py3" | awk '{print $1}'`
