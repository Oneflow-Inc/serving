ARG TRITON_VERSION=21.10

FROM nvcr.io/nvidia/tritonserver:$TRITON_VERSION-py3

# Uncomment it if you are in China
# RUN sed -i 's/security.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
#     sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list

RUN apt update && \
    apt -o Acquire::http::proxy=false install -y apt-utils software-properties-common && \
    apt update && \
    apt -o Acquire::http::proxy=false install -y git python3 python3-dev build-essential autoconf automake libtool make gcc g++ curl wget tar ccache rsync libopenblas-dev nasm python3-pip libssl-dev libonig-dev zlib1g-dev libboost-all-dev libre2-dev libb64-dev rapidjson-dev ninja-build libjpeg-dev && \
    apt clean

RUN pip3 install cmake
