conda create -n openvino python=3.9
conda activate openvino
echo "conda activate openvino" >> ~/.bash_profile
pip install openvino==2023.0.0 openvino-dev==2023.0.0 openvino-dev[onnx,pytorch]==2023.0.0
pip install opencv-python==4.5.5.64 opencv-contrib-python==4.5.5.64
pip install protobuf==3.20.2
"""
#报错
    from cv2 import *
ImportError: numpy.core.multiarray failed to import
pip install numpy==1.19.5
后来安装openvion又安装会最新的1.26，不再报错
"""

wget -c https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.0/macos/m_openvino_toolkit_macos_10_15_2023.0.0.10926.b4452d56304_x86_64.tgz
ln -s m_openvino_toolkit_macos_10_15_2023.0.0.10926.b4452d56304_x86_64 openvino
ls openvino/
echo "source /Volumes/data/openvino/setupvars.sh" >> ~/.bash_profile
source /Volumes/data/openvino/setupvars.sh

python -c '
from openvino.inference_engine import IECore
ie = IECore()
for device in ie.available_devices:
    print("\tDevice: {}".format(device))
'
#Device: CPU

