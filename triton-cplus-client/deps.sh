apt-get install -y build-essential autoconf libtool pkg-config
apt-get install -y cmake unzip
apt-get install -y clang libc++-dev
apt-get install -y libgflags-dev libgtest-dev
apt-get install -y libcurl4-openssl-dev

apt-get install -y libprotobuf-dev protobuf-compiler
:<<EOF
#grpc编译报错
/triton-cplus-client/grpc-1.60.0/src/compiler/config_protobuf.h:25:10: fatal error: google/protobuf/compiler/code_generator.h: No such file or directory
   25 | #include <google/protobuf/compiler/code_generator.h>
EOF
#client例子的cmakefile findpackage有错，apt安装的可以，但是grpc make时又在报错，只好2个都装一个给例子用，一个给grpc编译用
wget -c https://github.com/protocolbuffers/protobuf/releases/download/v3.20.3/protobuf-cpp-3.20.3.tar.gz
apt install -y autoconf
tar xzvf protobuf-cpp-3.20.3.tar.gz
cd protobuf-3.20.3
./autogen.sh
./configure --prefix=/usr/local/protobuf
make
make check
make install
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/protobuf/lib/pkgconfig

wget -c https://github.com/census-instrumentation/opencensus-proto/archive/v0.3.0.tar.gz
gzip opencensus-proto-0.3.0.tar

wget -c https://github.com/c-ares/c-ares/archive/refs/tags/cares-1_19_1.tar.gz
tar xzvf cares-1_19_1.tar.gz
cd c-ares-cares-1_19_1
cmake .
make
make install

wget -c https://github.com/abseil/abseil-cpp/archive/refs/tags/20230802.1.tar.gz
tar xvf abseil-cpp-20230802.1.tar
cd abseil-cpp-20230802.1
cmake .
make
make install

wget -c https://github.com/grpc/grpc/archive/refs/tags/v1.60.0.tar.gz
tar xvf grpc-1.60.0.tar
cd grpc-1.60.0
mkdir build
cd build
\cp ../../opencensus-proto-0.3.0.tar.gz http_archives/opencensus-proto-0.3.0/src.tar.gz
rm -rf ../third_party/cares/care
ln -s /triton-cplus-client/c-ares-cares-1_19_1 /triton-cplus-client/grpc-1.60.0/third_party/cares/cares
ccmake ..
#把absl的构建从module修改为package
cmake ..
make
make install