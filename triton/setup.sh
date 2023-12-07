#retry "" 20 1

version=r22.11
image_version=22.11
prjs=(server backend client)
echo "version:${version}"
for prj in ${prjs[*]}
do
  echo "prj:${prj}"
  retry "git clone -b ${version} https://github.com/triton-inference-server/${prj}.git ${prj}-${version}" 10 1
done

prjs=(dali onnxruntime python tensorflow tensorrt fil openvino pytorch stateful)
echo "version:${version}"
for prj in ${prjs[*]}
do
  echo "prj:${prj}"
  retry "git clone -b ${version} https://github.com/triton-inference-server/${prj}_backend.git ${prj}_backend-${version}" 10 1
done

tritonserver --model-repository=/data0/shouxieai/triton

:<<EOF
1, 从代码buid.py查看版本对应的内部版本号
https://github.com/triton-inference-server/server/blob/main/build.py
https://github.com/triton-inference-server/server/blob/v2.27.0/build.py

2，从内部版本号查看支持的cuda/cudnn/tensorRT版本
https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html#framework-matrix-2022
EOF

#tensorflow/tensorRT都不支持12以上的cuda，放弃升级cuda/cudnn，用指定本地源码而非container方式重新编译pytorch_backend
#https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=CentOS&target_version=7&target_type=runfile_local
#wget -c https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda_12.3.0_545.23.06_linux.run
#wget -c https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run
wget -c https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
#卸载旧版本
#ncurses图形界面卸载，逐个主机
cd /usr/local/cuda
bin/cuda-uninstaller
cd -
:<<EOF
#cuda 12+安装nvidia-fs需要mofed，infiniband的驱动，安装复杂放弃
yum install -y dkms mofed
wget https://www.mellanox.com/downloads/ofed/MLNX_OFED-5.4-1.0.3.0/MLNX_OFED_LINUX-5.4-1.0.3.0-rhel7.9-x86_64.tgz
EOF
#sh cuda_12.3.0_545.23.06_linux.run
#sh cuda_12.2.2_535.104.05_linux.run
sh cuda_11.8.0_520.61.05_linux.run

#可能会安装545.23.06版驱动覆盖原来525.85.05驱动，不确定3060/3050是否支持这样高版本，要取消，其他都保持选择尤其是Toolkit/Samples，CUDA编程需要
#取消nvidia_fs安装
nvcc -V
cat /proc/driver/nvidia/version
nvidia-smi
#如果无显示，重新安装driver
nvidia-smi
cudapath=/usr/local/cuda
ls -l /usr/local

#安装cudnn
#https://developer.nvidia.com/rdp/cudnn-download
rm -f cuda
:<<EOF
wget -c https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.6/local_installers/12.x/cudnn-linux-x86_64-8.9.6.50_cuda12-archive.tar.xz
xz -d cudnn-linux-x86_64-8.9.6.50_cuda12-archive.tar.xz
tar -xvf cudnn-linux-x86_64-8.9.6.50_cuda12-archive.tar
ln -s cudnn-linux-x86_64-8.9.6.50_cuda12-archive cuda
EOF
wget -c https://developer.nvidia.com/compute/cudnn/secure/8.6.0/local_installers/11.8/cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
xz -d cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
tar -xvf cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar
ln -s cudnn-linux-x86_64-8.6.0.163_cuda11-archive cuda
mv cuda/lib cuda/lib64
#cp -r /usr/local/cuda/include /usr/local/cuda/include.bk4-cudnn-setup
\cp cuda/include/cudnn*.h /usr/local/cuda/include
#cp -r /usr/local/cuda/lib64 /usr/local/cuda/lib64.bk4-cudnn-setup
\cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

#安装t
wget -c https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.5.1/tars/TensorRT-8.5.1.7.Linux.x86_64-gnu.cuda-11.8.cudnn8.6.tar.gz
rm -f /data0/trt
tar xzvf TensorRT-8.5.1.7.Linux.x86_64-gnu.cuda-11.8.cudnn8.6.tar.gz
ln -s TensorRT-8.5.1.7 trt

:<<EOF
conda create -n triton python=3.8 -y
conda activate triton

wget -c https://download.pytorch.org/whl/cu121/torch-2.1.0%2Bcu121-cp38-cp38-linux_x86_64.whl#sha256=4c83190ad649c77adaf6e1c616998f10598db696912ea7a410831632890b49bf
wget -c https://download.pytorch.org/whl/cu121/torchvision-0.16.0%2Bcu121-cp38-cp38-linux_x86_64.whl#sha256=967c0f8ada2abeed430b33e83ce7b3de5233675c8b020c18c1255dca065b4add
#torchtext没有gpu版本
pip install torch-2.1.0+cu121-cp38-cp38-linux_x86_64.whl
pip install torchvision-0.16.0+cu121-cp38-cp38-linux_x86_64.whl
#pip install tensorflow==2.13.1
pip install tensorflow==2.15.0rc1
pip install tensorflow-probability==0.21.0
EOF

python <<EOP
import tensorrt as trt
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))
import torch
print('torch.cuda.device_count():', torch.cuda.device_count())
print('torch.cuda.get_device_name(0):', torch.cuda.get_device_name(0))
EOP


#在centos上编译失败，因为re2和absl的版本配套关系（re2似乎是Protobuf需要）总是失败，这2个都调整到2023年初的版本后解决
#triton
#git clone https://github.com/Microsoft/vcpkg.git
#cd vcpkg
#./bootstrap-vcpkg.sh
#echo "export PATH=\$PATH:/data0/vcpkg" >> ~/.bashrc
#./vcpkg integrate install
#vcpkg install rapidjson
git clone https://github.com/Tencent/rapidjson.git
cd rapidjson
mkdir build
cd build
cmake ..
make install
boost_version=1.83.0
boost_version_file=1_83_0
wget -c https://boostorg.jfrog.io/artifactory/main/release/${boost_version}/source/boost_${boost_version_file}.tar.gz
tar xzvf boost_${boost_version_file}.tar.gz
cd boost_${boost_version_file}
./bootstrap
./b2
./b2 install
absl_version=20230125.3
wget -c https://github.com/abseil/abseil-cpp/archive/refs/tags/${absl_version}.tar.gz
tar xzvf abseil-cpp-${absl_version}.tar.gz
cd abseil-cpp-${absl_version}
mkdir build
cd build
cmake ..
make install
:<<EOF
rm -f /usr/local/lib64/pkgconfig/libabsl_*
rm -f /usr/local/lib64/pkgconfig/absl_*
rm -rf /usr/local/include/absl
rm -rf /usr/local/lib64/cmake/absl
EOF
re2_version=2023-02-01
https://github.com/google/re2/archive/refs/tags/${re2_version}.tar.gz
tar xzvf re2-${re2_version}.tar.gz
cd re2-${re2_version}
mkdir build
cd build
cmake ..
make install
:<<EOF
rm -f /usr/local/lib64/libre2.a
rm -f /usr/local/lib64/pkgconfig/re2.pc
rm -rf /usr/local/include/re2
rm -rf /usr/local/lib64/cmake/re2
EOF
#yum install re2-devel
#只有.so没有.a，还是没法编译triton
dnf install dnf-plugin-config-manager -y
dnf config-manager \
    --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
dnf clean expire-cache \
    && dnf install -y datacenter-gpu-manager
yum install numactl-devel*x86_64 -y
git clone https://github.com/BuLogics/libb64.git
cd libb64
cp -r include/b64 /usr/local/include/
cd src
make
cp libb64.a /usr/local/lib/
:<<EOF
rm -f /usr/local/lib/libb64.a
rm -rf /usr/local/include/b64
EOF
#triton_version=2.39.0
#triton_version=2.29.0
triton_version=2.27.0
wget -c https://github.com/triton-inference-server/server/archive/refs/tags/v${triton_version}.tar.gz
tar xzvf server-${triton_version}.tar.gz
cd server-${triton_version}
#git clone --recursive https://github.com/triton-inference-server/server.git
#cd server
#mkdir build
#cd build
#cmake -DCMAKE_INSTALL_PREFIX:PATH=/data0/triton ..
#cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DRapidJSON_DIR=/data0/rapidjson ..
#make install
./build.py -v --no-container-build --build-dir=/data0/triton --enable-all
#

git clone --recursive https://github.com/triton-inference-server/backends.git
cd backends
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=/data0/triton ..
make install
cd examples/backends
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=/data0/triton ..
make install
#

git clone --recursive https://github.com/triton-inference-server/client.git
cd client
mkdir build
cd build
file=../src/c++/perf_analyzer/client_backend/torchserve/../../client_backend/client_backend.h
cp ${file} ${file}.bk
sed -i "/#include <vector>/a\#include <unordered_map>" ${file}
#sourceforge/github上下载的libb64都是需要手工copy的include/b64和libb64.a，编译时会提示BUFFERSIZE常量未定义错误
wget -c http://www6.atomicorp.com/channels/atomic/centos/7/x86_64/RPMS/libb64-libs-1.2.1-2.1.el7.art.x86_64.rpm
rpm -Uvh libb64-libs-1.2.1-2.1.el7.art.x86_64.rpm
wget -c http://www6.atomicorp.com/channels/atomic/centos/7/x86_64/RPMS/libb64-1.2.1-2.1.el7.art.x86_64.rpm
rpm -Uvh libb64-1.2.1-2.1.el7.art.x86_64.rpm
wget -c http://www6.atomicorp.com/channels/atomic/centos/7/x86_64/RPMS/libb64-devel-1.2.1-2.1.el7.art.x86_64.rpm
rpm -Uvh libb64-devel-1.2.1-2.1.el7.art.x86_64.rpm
wget -c https://github.com/grpc/grpc/archive/refs/tags/v1.60.0.tar.gz
tar xzvf grpc-1.60.0.tar.gz
cd grpc-1.60.0
mkdir build
cd build
cmake ..
file=../src/c++/CMakeLists.txt
cp ${file} ${file}.bk
sed -i '/project(cc-clients LANGUAGES C CXX)/a\set(CMAKE_CXX_STANDARD 17)' ${file}
pip install grpcio
pip install grpcio-tools
echo "export PATH=\$PATH:/data0/maven/bin" >> ~/.bashrc
cd src/python/libray
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=/data0/triton
make install
pip install /data0/triton/python/tritonclient-0.0.0-py3-none-manylinux1_x86_64.whl
#fq
#！！！注释掉/etc/hosts内github
#,
#    "http://hub-mirror.c.163.com"
docker stop centos7-netutil-ccplus7-go-jdk
docker rm centos7-netutil-ccplus7-go-jdk
#苹果共享id：f54@falemon.com/cM9WZ1d0uQ
#--privileged --net host
docker run --net host -itd --name centos7-netutil-ccplus7-go-jdk --restart unless-stopped -v /Volumes/data/workspace/shouxieai/dockerhost:/host harbor.my.org:1080/base/python:3.8-centos7-netutil-ccplus7-go-jdk tail -f /dev/null
docker exec -it centos7-netutil-ccplus7-go-jdk bash
  git clone https://github.com/grpc/grpc.git
  #比较host/container下载速度，应该都能到M以上
  scl enable devtoolset-10 bash
  cp /host/retry /usr/bin
  cd /host/client-r22.11
  mkdir build
  cd build
  #cmake -DCMAKE_INSTALL_PREFIX:PATH=/../client-built -DTRITON_ENABLE_GPU=on -DTRITON_ENABLE_CC_HTTP=on -DTRITON_ENABLE_CC_GRPC=on -DTRITON_ENABLE_PERF_ANALYZER=on -DTRITON_ENABLE_PYTHON_GRPC=on ..
  cmake -DCMAKE_INSTALL_PREFIX:PATH=/host/client-built -DTRITON_ENABLE_GPU=on -DTRITON_ENABLE_CC_HTTP=on -DTRITON_ENABLE_CC_GRPC=on -DTRITON_ENABLE_PERF_ANALYZER=on -DTRITON_ENABLE_PYTHON_GRPC=on ..
  retry "make cc-clients python-clients java-clients" 10 1
  cd /host/client-r22.11/build && watch du -h --max-depth=1
  #./_deps大概在2-4s增加1M，说明在正常下载依赖github源码
  #host也无法clone grpc以后，换一个站点重连，换了多个还不行就把客户端退出重新再进入
#总是停在Cloning into "grpc"，grpc下载编译依赖太多报错

yum install libarchive-devel -y
onnxruntime_version=1.16.1
wget -c https://github.com/microsoft/onnxruntime/releases/download/v${onnxruntime_version}/onnxruntime-linux-x64-gpu-${onnxruntime_version}.tgz
tar xzvf onnxruntime-linux-x64-gpu-${onnxruntime_version}.tgz
ln -s onnxruntime-linux-x64-gpu-${onnxruntime_version} onnxruntime
arr=(python pytorch tensorrt onnxruntime openvino tensorflow stateful)
for be in ${arr[*]}
do
  echo "DEBUG >>>>>> be:${be}"
  bename=${be}_backend
  echo "DEBUG >>>>>> bename:${bename}"
  git clone --recursive https://github.com/triton-inference-server/${bename}.git
  cd ${bename}
  mkdir build
  cd build
  if [[ "${be}" == "python" ]]; then
    cmake -DTRITON_ENABLE_GPU=ON \
    -DTRITON_BACKEND_REPO_TAG=r22.11 \
    -DTRITON_COMMON_REPO_TAG=r22.11 \
    -DTRITON_CORE_REPO_TAG=r22.11 \
    -DCMAKE_INSTALL_PREFIX:PATH=/data/triton \
     ..
  fi
  #done

  if [[ "${be}" == "pytorch" ]]; then
    yum install -y patchelf
    file=../CMakeLists.txt
    cp ${file} ${file}.bk
    sed -i "s/cxx_std_11/cxx_std_17/g" ${file}
    #cmake -DCMAKE_INSTALL_PREFIX:PATH=/data0/triton -DTRITON_PYTORCH_DOCKER_IMAGE="nvcr.io/nvidia/pytorch:23.10-py3" ..
    #cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_PYTORCH_INCLUDE_PATHS="<PATH_PREFIX>/torch;<PATH_PREFIX>/torch/torch/csrc/api/include;<PATH_PREFIX>/torchvision" -DTRITON_PYTORCH_LIB_PATHS="<LIB_PATH_PREFIX>" ..
    #cmake -DCMAKE_INSTALL_PREFIX:PATH=/data0/triton -DTRITON_PYTORCH_INCLUDE_PATHS="/data0/envs/deepspeed/lib/python3.9/site-packages/torch;/data0/envs/deepspeed/lib/python3.9/site-packages/torch/include/torch/csrc/api/include;/data0/envs/deepspeed/lib/python3.9/site-packages/torchvision" -DTRITON_PYTORCH_LIB_PATHS="/data0/envs/deepspeed/lib/python3.9/site-packages/torch/lib;/data0/envs/deepspeed/lib/python3.9/site-packages/torchvision.libs" ..
    #总是提示找不到torchvision，enable TRT以后又提示找不到trt
    #cmake -DCMAKE_INSTALL_PREFIX:PATH=/data0/triton -DTRITON_PYTORCH_DOCKER_IMAGE="nvcr.io/nvidia/pytorch:23.10-py3" ..
    #nerdctl pull nvcr.io/nvidia/pytorch:${image_version}-py3
    #retry "cmake -DCMAKE_INSTALL_PREFIX:PATH=/data0/triton -DTRITON_PYTORCH_DOCKER_IMAGE=\"nvcr.io/nvidia/pytorch:${image_version}-py3\" .." 20 1
    #有libc10版本匹配问题，因为container是用ubuntu做的，没法用container方式编译
    conda create -n triton python=3.8 -y
    #conda remove -n triton --all -y
    conda activate triton
    pip install torch-1.13.0+cu117.with.pypi.cudnn-cp38-cp38-linux_x86_64.whl
    pip install torch-1.13.0+cpu-cp38-cp38-linux_x86_64.whl

    wget -c https://download.pytorch.org/libtorch/cu117/libtorch-shared-with-deps-1.13.0%2Bcu117.zip
    unzip libtorch-shared-with-deps-1.13.0+cu117.zip

    wget -c https://github.com/pytorch/vision/archive/refs/tags/v0.14.0.tar.gz
    tar xzvf vision-0.14.0.tar.gz
    cd vision-0.14.0
    mkdir build
    cd build
    export Torch_DIR=/data0/envs/triton/lib/python3.8/site-packages/torch/share/cmake
    retry "cmake -DCMAKE_INSTALL_PREFIX:PATH=/data0/triton -DUSE_PYTHON=on -DWITH_CUDA=on .." 10 1
    cp /data0/triton/lib64/libtorchvision.so /data0/envs/triton/lib/python3.8/site-packages/torch/lib/

    #pip install torchvision-0.14.0+cu117-cp38-cp38-linux_x86_64.whl
    retry "cmake -DCMAKE_INSTALL_PREFIX:PATH=/data0/triton -DTRITON_PYTORCH_INCLUDE_PATHS=/data0/envs/triton/lib/python3.8/site-packages/torch/include;/data0/libtorch/include -DTRITON_PYTORCH_LIB_PATHS=/data0/envs/triton/lib/python3.8/site-packages/torch/lib .." 10 1
:<<EOF
#报错缺少文件，下载libtorch也没有fuser目录
/data0/triton-src/pytorch_backend-r22.11/src/libtorch_utils.h:37:10: fatal error: torch/csrc/jit/codegen/fuser/interface.h: No such file or directory
   37 | #include <torch/csrc/jit/codegen/fuser/interface.h>
      |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
EOF
  fi
  #done

  if [[ "${be}" == "tensorrt" ]]; then
    file=../CMakeLists.txt
    cp ${file} ${file}.bk
    #sed -i "s/c++11/c++17/g" ${file}
    sed -i "s/ -Werror//g" ${file}
    cmake -DCMAKE_INSTALL_PREFIX:PATH=/data0/triton -DTRITON_ENABLE_CC_HTTP=ON -DTRITON_ENABLE_CC_GRPC=ON -DTRITON_ENABLE_PERF_ANALYZER=ON -DTRITON_ENABLE_PERF_ANALYZER_C_API=ON -DTRITON_ENABLE_PERF_ANALYZER_TFS=ON -DTRITON_ENABLE_PERF_ANALYZER_TS=ON -DTRITON_ENABLE_PYTHON_HTTP=ON -DTRITON_ENABLE_PYTHON_GRPC=ON -DTRITON_ENABLE_JAVA_HTTP=ON -DTRITON_ENABLE_GPU=ON -DTRITON_ENABLE_EXAMPLES=ON -DTRITON_ENABLE_TESTS=ON \
    -DTRITON_TENSORRT_LIB_PATHS=/data0/trt/lib \
    -DTRITON_TENSORRT_INCLUDE_PATHS=/data0/trt/include \
    -DNVINFER_LIBRARY=/data0/trt/lib/libnvinfer_static.a \
    -DNVINFER_PLUGIN_LIBRARY=/data0/trt/lib/libnvinfer_plugin_static.a \
    ..
  fi
  #done

  if [[ "${be}" == "onnxruntime" ]]; then
    cmake -DCMAKE_INSTALL_PREFIX:PATH=/data0/triton -DTRITON_ENABLE_ONNXRUNTIME_TENSORRT=ON -DTRITON_ENABLE_ONNXRUNTIME_OPENVINO=ON \
    -DTRITON_BUILD_ONNXRUNTIME_VERSION=1.16.0 \
    -DTRITON_BUILD_CONTAINER_VERSION=23.10 \
    -DTRITON_BUILD_ONNXRUNTIME_OPENVINO_VERSION=2023.0.0 \
    -DCUDA_SDK_ROOT_DIR=/usr/local/cuda \
    -DTRITON_ONNXRUNTIME_INCLUDE_PATHS=/data0/onnxruntime/include \
    -DTRITON_ONNXRUNTIME_LIB_PATHS=/data0/onnxruntime/lib \
    -DOV_LIBRARY=/data0/openvino/runtime/lib/intel64 \
    -DTRITON_BUILD_CONTAINER=OFF \
    ..
  fi
  #done

  if [[ "${be}" == "openvino" ]]; then
    cmake -DCMAKE_INSTALL_PREFIX:PATH=/data0/triton \
    -DTRITON_BUILD_CONTAINER_VERSION=23.10 \
    -DTRITON_BUILD_OPENVINO_VERSION=2023.0.0 \
    ..
  fi
  #make install

  if [[ "${be}" == "tensorflow" ]]; then
    cmake -DCMAKE_INSTALL_PREFIX:PATH=/data0/triton -DTRITON_TENSORFLOW_DOCKER_IMAGE="nvcr.io/nvidia/tensorflow:23.10-tf2-py3" ..
  fi
  #done

  if [[ "${be}" == "stateful" ]]; then
    cd ..
    file=NGC_VERSION
    cp ${file} ${file}.bk
    echo "23.10" > ${file}
    python3 ./build.py
  fi
  #

  if [[ "${be}" != "stateful" ]]; then
    make install
    cd ../../
  fi
done
ln -s /data0/triton /opt/triton
echo "export PATH=\$PATH:/opt/triton/bin" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/triton/lib64" >> ~/.bashrc






















































...
..........
.0000