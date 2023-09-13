#centos 7, dtpct
mkdir /data0/shouxieai
conda create -n shouxieai python=3.9
conda activate shouxieai
cd /data0
pip install torch-1.12.0+cu116-cp39-cp39-linux_x86_64.whl
pip install torchvision-0.13.0+cu116-cp39-cp39-linux_x86_64.whl
pip install torchtext-0.13.0-cp39-cp39-linux_x86_64.whl
pip install onnxruntime onnx opencv-python
yum install -y libX11
yum install -y libXext