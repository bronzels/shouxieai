git clone https://github.com/nlohmann/json.git
cd json
mkdir build
cd build/
cmake ..
make
make install

#opencv依赖的gflag安装用google的namespace，但是openvino用gflag
gflags_version=2.2.2
wget -c https://github.com/gflags/gflags/archive/v${gflags_version}.tar.gz -O gflags-${gflags_version}.tar.gz  #下载源码
tar xzvf gflags-${gflags_version}.tar.gz
cd gflags-${gflags_version}
mkdir build && cd build  #建立编译文件夹，用于存放临时文件
#opencv, -DCMAKE_INSTALL_PREFIX=/usr -DGFLAGS_NAMESPACE=gflags
#openvino, -DCMAKE_INSTALL_PREFIX=/usr/local/gflags -DGFLAGS_NAMESPACE=google
cmake .. -DCMAKE_CXX_FLAGS="-fPIC" -DCMAKE_INSTALL_PREFIX=/usr/local/gflags -DBUILD_SHARED_LIBS=ON -DGFLAGS_NAMESPACE=gflags -G "Unix Makefiles"  #使用 cmake 进行动态编译生成 Makefile 文件，安装路径为/usr
make # make 编译
make install # 安装库

pugixml_version=1.14
wget -c https://github.com/zeux/pugixml/releases/download/v${pugixml_version}/pugixml-${pugixml_version}.tar.gz
tar xzvf pugixml-${pugixml_version}.tar.gz
cd pugixml-${pugixml_version}
mkdir build && cd build
cmake ..
make
make install
cmake .. -DBUILD_SHARED_LIBS=ON
make
make install

cp -r /data0/openvino/samples/cpp/common ./
cmake .
make
cp ../openvino/alexnet.xml ./
cp ../openvino/alexnet.bin ./
cp ../openvino/car.jpeg ./
./samples_hello_classification alexnet.xml car.jpeg CPU

#13.2一开始编译成功，但是编译时自己的头文件报很多错误，后来编译中间error，但是具体什么原因不知道
#放弃
#gcc 13.2 to support C++20/23
gcc_version=13.2.0
wget -c http://ftp.gnu.org/gnu/gcc/gcc-${gcc_version}/gcc-${gcc_version}.tar.gz
tar xzvf gcc-${gcc_version}.tar.gz
cd gcc-${gcc_version}
./contrib/download_prerequisites
mkdir build && cd build
../configure --prefix=/usr/local/gcc-13.2.0 -enable-checking=release -enable-languages=c,c++ -disable-multilib
make -j24
make install
:<<EOF
#source scl_source enable devtoolset-7
#source scl_source enable devtoolset-10
#source scl_source enable devtoolset-11
#gcc 13.2
export PATH=$PATH:/usr/local/gcc-13.2.0/bin
EOF
sed -i 's/source scl_source enable/#source scl_source enable/g' ~/.bashrc
echo "export PATH=\$PATH:/usr/local/gcc-13.2.0/bin" >> ~/.bashrc

