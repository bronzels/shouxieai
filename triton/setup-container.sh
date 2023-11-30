
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  tee /etc/yum.repos.d/nvidia-container-toolkit.repo

yum install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=containerd
systemctl restart containerd
nerdctl run --rm --runtime=nvidia-container-runtime --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi

wget -c https://developer.download.nvidia.com/compute/redist/nvidia-dali-cuda110/nvidia_dali_cuda110-1.31.0-10168358-py3-none-manylinux2014_x86_64.whl

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
wget -c https://github.com/triton-inference-server/server/releases/download/v2.27.0/v2.27.0_ubuntu2004.clients.tar.gz
mkdir ../client-v2.27.0
cd ../client-v2.27.0
tar xzvf ../v2.27.0_ubuntu2004.clients.tar.gz
cd -
nerdctl exec -it `nerdctl ps -a | grep tritonserver:${triton_version}-py3 | grep "8000->8000" | awk '{print $1}'` /bin/bash
  cd /models
  pip install torch-1.12.0+cu116-cp38-cp38-linux_x86_64.whl
  pip install torchvision-0.13.0+cu116-cp38-cp38-linux_x86_64.whl
  pip install tritonclient[all]
  cd /client/python
  pip3 install --upgrade tritonclient-2.27.0-py3-none-manylinux1_x86_64.whl[all]
  pip install onnxruntime==1.13.1
nerdctl exec `nerdctl ps -a | grep tritonserver:${triton_version}-py3-tchtf | grep 8000 | awk '{print $1}'` mkdir /opt/tritonserver/backends/minimal
nerdctl cp /data0/backend/examples/backends/minimal/libtriton_minimal.so `nerdctl ps -a | grep tritonserver:${triton_version}-py3-tchtf | awk '{print $1}'`:/opt/tritonserver/backends/minimal
nerdctl exec `nerdctl ps -a | grep tritonserver:${triton_version}-py3-tchtf | awk '{print $1}'` ls /opt/tritonserver/backends/minimal
nerdctl commit `nerdctl ps -a | grep tritonserver:${triton_version}-py3 | grep "8000->8000" | awk '{print $1}'` tritonserver:${triton_version}-py3-tchtf
nerdctl run --gpus all --shm-size=1g --rm -p8000:8000 -p8001:8001 -p8002:8002 --net=host --rm -v ${PWD}:/models tritonserver:${triton_version}-py3-tchtf tritonserver --model-repository=/models

#--strict-model-config=false
#--model-control-mode poll
#--model-control-mode explicit
#--exit-on-error=false

nerdctl exec -it `nerdctl ps -a | grep tritonserver:${triton_version}-py3-tchtf | grep 8000 | awk '{print $1}'` /bin/bash

wget -c https://s3.jcloud.sjtu.edu.cn/899a892efef34b1b944a19981040f55b-oss01/pytorch-wheels/cu116/torch-1.12.0+cu116-cp38-cp38-linux_x86_64.whl
wget -c https://mirror.sjtu.edu.cn/pytorch-wheels/cu116/torchvision-0.13.0+cu116-cp38-cp38-linux_x86_64.whl
nerdctl run -it --net=host --rm -v /usr/local/cuda:/usr/local/cuda -v ${PWD}:/models nvcr.io/nvidia/tritonserver:${triton_version}-py3-sdk /bin/bash
  cd /models
  pip install torch-1.12.0+cu116-cp38-cp38-linux_x86_64.whl
  pip install torchvision-0.13.0+cu116-cp38-cp38-linux_x86_64.whl
  pip install tensorflow==2.8.0
  pip install tensorflow-probability==0.16.0
  pip install validators==0.22.0
  pip install openvino-dev==2022.1.0
  pip install nvidia-ml-py
  pip install nvidia-pyindex
  pip install nvidia_dali_cuda110-1.31.0-10168358-py3-none-manylinux2014_x86_64.whl
nerdctl commit `nerdctl ps -a | grep tritonserver:${triton_version}-py3-sdk | awk '{print $1}'` tritonserver:${triton_version}-py3-sdk-tchtf
nerdctl stop `nerdctl ps -a | grep tritonserver:${triton_version}-py3-sdk | awk '{print $1}'`
nerdctl rm `nerdctl ps -a | grep tritonserver:${triton_version}-py3-sdk | awk '{print $1}'`

nerdctl exec -it `nerdctl ps -a | grep tritonserver:${triton_version}-py3-sdk | awk '{print $1}'` /bin/bash

#nerdctl run -it --net=host --rm -v `realpath $PWD/../data`:/data -v ${PWD}:/models tritonserver:${triton_version}-py3-sdk-tchtf /bin/bash
nerdctl run --runtime=nvidia-container-runtime -it --net=host --rm -v /usr/local/cuda:/usr/local/cuda -v `realpath $PWD/../data`:/data -v ${PWD}:/models tritonserver:${triton_version}-py3-sdk-tchtf /bin/bash
  cd /models

nerdctl stop `nerdctl ps -a | grep "tritonserver:${triton_version}-py3-tchtf" | awk '{print $1}'`
nerdctl restart `nerdctl ps -a | grep "tritonserver:${triton_version}-py3-tchtf" | awk '{print $1}'`
nerdctl logs -f `nerdctl ps -a | grep "tritonserver:${triton_version}-py3-tchtf" | awk '{print $1}'`

nerdctl start `nerdctl ps -a | grep "tritonserver:${triton_version}-py3" | awk '{print $1}'`

nerdctl rm `nerdctl ps -a | grep Exited | awk '{print $1}'`

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