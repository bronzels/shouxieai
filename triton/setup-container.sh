#triton_version=22.04
triton_version=22.11
echo "export triton_version=${triton_version}" >> ~/.bashrc
nerdctl pull nvcr.io/nvidia/tritonserver:${triton_version}-py3
nerdctl pull nvcr.io/nvidia/tritonserver:${triton_version}-py3-sdk

git clone https://github.com/triton-inference-server/backend.git
cd backend/examples/backends/minimal
file=src/minimal.cc
cp ${file} ${file}.bk

cmake . -DTRITON_COMMON_REPO_TAG=r22.11 -DTRITON_CORE_REPO_TAG=r22.11 -DTRITON_BACKEND_REPO_TAG=r22.11
make

nerdctl run --gpus all --shm-size=1g --rm -p8000:8000 -p8001:8001 -p8002:8002 --net=host --rm -v ${PWD}:/models nvcr.io/nvidia/tritonserver:${triton_version}-py3 tritonserver --model-repository=/models
nerdctl exec -it `nerdctl ps -a | grep tritonserver:${triton_version}-py3 | grep "8000->8000" | awk '{print $1}'` /bin/bash
  cd /models
  pip install torch-1.12.0+cu116-cp38-cp38-linux_x86_64.whl
  pip install torchvision-0.13.0+cu116-cp38-cp38-linux_x86_64.whl
nerdctl exec `nerdctl ps -a | grep tritonserver:${triton_version}-py3-tchtf | awk '{print $1}'` mkdir /opt/tritonserver/backends/minimal
nerdctl cp /data0/backend/examples/backends/minimal/libtriton_minimal.so `nerdctl ps -a | grep tritonserver:${triton_version}-py3-tchtf | awk '{print $1}'`:/opt/tritonserver/backends/minimal
nerdctl exec `nerdctl ps -a | grep tritonserver:${triton_version}-py3-tchtf | awk '{print $1}'` ls /opt/tritonserver/backends/minimal
nerdctl commit `nerdctl ps -a | grep tritonserver:${triton_version}-py3 | grep "8000->8000" | awk '{print $1}'` tritonserver:${triton_version}-py3-tchtf
nerdctl run --gpus all --shm-size=1g --rm -p8000:8000 -p8001:8001 -p8002:8002 --net=host --rm -v ${PWD}:/models tritonserver:${triton_version}-py3-tchtf tritonserver --model-repository=/models

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

nerdctl stop `nerdctl ps -a | grep "tritonserver:${triton_version}-py3-tchtf" | awk '{print $1}'`
nerdctl restart `nerdctl ps -a | grep "tritonserver:${triton_version}-py3-tchtf" | awk '{print $1}'`
nerdctl logs -f `nerdctl ps -a | grep "tritonserver:${triton_version}-py3-tchtf" | awk '{print $1}'`

nerdctl start `nerdctl ps -a | grep "tritonserver:${triton_version}-py3" | awk '{print $1}'`

"""
tensorflow要求的numpy版本高，openvino的版本低，有冲突
Installing collected packages: numpy
  Attempting uninstall: numpy
    Found existing installation: numpy 1.19.5
    Uninstalling numpy-1.19.5:
      Successfully uninstalled numpy-1.19.5
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
openvino 2022.1.0 requires numpy<1.20,>=1.16.6, but you have numpy 1.24.4 which is incompatible.
openvino-dev 2022.1.0 requires numpy<1.20,>=1.16.6, but you have numpy 1.24.4 which is incompatible.
openvino-dev 2022.1.0 requires numpy<=1.21,>=1.16.6; python_version > "3.6", but you have numpy 1.24.4 which is incompatible.
numba 0.56.4 requires numpy<1.24,>=1.18, but you have numpy 1.24.4 which is incompatible.
Successfully installed numpy-1.24.4
"""