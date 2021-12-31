FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

# Uncomment it if you are in China
RUN sed -i 's/security.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list
RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list

# Add common tools available in apt repository. We choose not to support python2
RUN apt update && \
    apt -o Acquire::http::proxy=false install -y apt-utils software-properties-common && \
    apt update && \
    apt -o Acquire::http::proxy=false install -y git python3 python3-dev build-essential autoconf automake libtool make gcc g++ curl wget tar ccache rsync libopenblas-dev nasm python3-pip libssl-dev libonig-dev zlib1g-dev libboost-all-dev libre2-dev libb64-dev rapidjson-dev ninja-build libjpeg-dev && \
    apt clean

# cmake, dcgm
RUN mkdir /environment && \
    cd /environment  && \
    wget -O cmake.tar.gz https://github.com/Kitware/CMake/releases/download/v3.21.4/cmake-3.21.4-linux-x86_64.tar.gz && \
    tar -xvf cmake.tar.gz
ENV PATH="/environment/cmake-3.21.4-linux-x86_64/bin/:${PATH}"
RUN cd /environment  && \
    wget https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu1804/x86_64/datacenter-gpu-manager_2.3.1_amd64.deb && \
    dpkg -i datacenter-gpu-manager_2.3.1_amd64.deb

# build tritonserver
RUN git clone https://github.com/triton-inference-server/server && \
    cd server && \
    git checkout r21.10 && \
    ./build.py --cmake-dir=$(pwd)/build --build-dir=/tmp/citritonbuild --enable-logging --enable-stats --enable-tracing --enable-metrics --enable-gpu-metrics --enable-gpu --endpoint=http --endpoint=grpc --repo-tag=common:r21.10 --repo-tag=core:r21.10 --repo-tag=backend:r21.10 --repo-tag=thirdparty:r21.10 --backend=ensemble --no-container-build

# build oneflow
RUN git clone https://github.com/Oneflow-Inc/oneflow --depth=1 && \
    cd oneflow && \
    mkdir build  && \
    cd build && \
    cmake .. -C ../cmake/caches/cn/fast/cuda-75.cmake -DBUILD_CPP_API=ON -DBUILD_MONOLITHIC_LIBONEFLOW_CPP_SO=ON -DBUILD_SHARED_LIBS=OFF -DWITH_MLIR=ON && \
    ninja

# build oneflow-backend, need authentication
RUN git clone https://github.com/Oneflow-Inc/ofserving  && \
    cd ofserving/oneflow-backend && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install  -DTRITON_BACKEND_REPO_TAG=r21.10 -DTRITON_CORE_REPO_TAG=r21.10 -DTRITON_COMMON_REPO_TAG=r21.10 -G Ninja -DCMAKE_PREFIX_PATH=/triton/oneflow/build-clang/liboneflow/share -DTRITON_ENABLE_GPU=ON .. && \
    ninja
