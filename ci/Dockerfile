FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

# Uncomment it if you are in China
RUN sed -i 's/security.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list

RUN apt update && \
    apt -o Acquire::http::proxy=false install -y apt-utils software-properties-common && \
    apt update && \
    apt -o Acquire::http::proxy=false install -y git python3 python3-dev build-essential autoconf automake libtool make gcc g++ curl wget tar ccache rsync libopenblas-dev nasm python3-pip libssl-dev libonig-dev zlib1g-dev libboost-all-dev libre2-dev libb64-dev rapidjson-dev ninja-build libjpeg-dev locales && \
    locale-gen en_US.UTF-8 && \
    apt clean
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'

# cmake, dcgm
RUN mkdir /tools && \
    cd /tools  && \
    wget -O cmake.tar.gz https://github.com/Kitware/CMake/releases/download/v3.21.4/cmake-3.21.4-linux-x86_64.tar.gz && \
    tar -xvf cmake.tar.gz && \
    wget https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu1804/x86_64/datacenter-gpu-manager_2.3.1_amd64.deb && \
    dpkg -i datacenter-gpu-manager_2.3.1_amd64.deb && \
    rm -rf datacenter-gpu-manager_2.3.1_amd64.deb && \
    rm -rf cmake.tar.gz
ENV PATH="/tools/cmake-3.21.4-linux-x86_64/bin/:${PATH}"

# build tritonserver
RUN git clone https://github.com/triton-inference-server/server && \
    cd server && \
    git checkout r21.10 && \
    ./build.py --cmake-dir=$(pwd)/build --build-dir=/tmp/citritonbuild --enable-logging --enable-stats --enable-tracing --enable-metrics --enable-gpu-metrics --enable-gpu --endpoint=http --endpoint=grpc --repo-tag=common:r21.10 --repo-tag=core:r21.10 --repo-tag=backend:r21.10 --repo-tag=thirdparty:r21.10 --backend=ensemble --no-container-build && \
    mkdir /opt/tritonserver && \
    cp -r /tmp/citritonbuild/tritonserver/install/* /opt/tritonserver/ && \
    rm -rf /tmp/citritonbuild
