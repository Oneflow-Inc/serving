
#
# Multistage build.
#
ARG TRITON_VERSION=2.15.0
ARG TRITON_CONTAINER_VERSION=21.10

FROM nvcr.io/nvidia/tritonserver:21.10-py3 as full
FROM nvcr.io/nvidia/tritonserver:21.10-py3-min

ARG TRITON_VERSION
ARG TRITON_CONTAINER_VERSION

ENV TRITON_SERVER_VERSION ${TRITON_VERSION}
ENV NVIDIA_TRITON_SERVER_VERSION ${TRITON_CONTAINER_VERSION}
LABEL com.nvidia.tritonserver.version="${TRITON_SERVER_VERSION}"

ENV PATH /opt/tritonserver/bin:${PATH}

ENV TF_ADJUST_HUE_FUSED         1
ENV TF_ADJUST_SATURATION_FUSED  1
ENV TF_ENABLE_WINOGRAD_NONFUSED 1
ENV TF_AUTOTUNE_THRESHOLD       2
ENV TRITON_SERVER_GPU_ENABLED   1

# Create a user that can be used to run triton as
# non-root. Make sure that this user to given ID 1000. All server
# artifacts copied below are assign to this user.
ENV TRITON_SERVER_USER=triton-server
RUN userdel tensorrt-server > /dev/null 2>&1 || true &&     if ! id -u $TRITON_SERVER_USER > /dev/null 2>&1 ; then         useradd $TRITON_SERVER_USER;     fi &&     [ `id -u $TRITON_SERVER_USER` -eq 1000 ] &&     [ `id -g $TRITON_SERVER_USER` -eq 1000 ]

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive

# Common dependencies. FIXME (can any of these be conditional? For
# example libcurl only needed for GCS?)
RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common libb64-0d libcurl4-openssl-dev libre2-5 git dirmngr libnuma-dev curl python3 python3-dev build-essential autoconf automake libtool make gcc g++ curl wget tar ccache rsync libopenblas-dev nasm python3-pip libssl-dev libonig-dev zlib1g-dev libboost-all-dev libre2-dev libb64-dev rapidjson-dev ninja-build libjpeg-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple && \
    pip3 install cmake

ENV DCGM_VERSION 2.2.9
# Install DCGM. Steps from https://developer.nvidia.com/dcgm#Downloads
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && \
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub && \
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" && \
    apt-get update && apt-get install -y datacenter-gpu-manager=1:2.2.9

# Extra defensive wiring for CUDA Compat lib
RUN ln -sf ${_CUDA_COMPAT_PATH}/lib.real ${_CUDA_COMPAT_PATH}/lib  && echo ${_CUDA_COMPAT_PATH}/lib > /etc/ld.so.conf.d/00-cuda-compat.conf  && ldconfig  && rm -f ${_CUDA_COMPAT_PATH}/lib

# Install TensorRT
RUN wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/tensorrt/nv-tensorrt-repo-ubuntu2004-cuda11.3-trt8.0.3.4-ga-20210831_1-1_amd64.deb && \
    dpkg -i nv-tensorrt-repo-ubuntu2004-cuda11.3-trt8.0.3.4-ga-20210831_1-1_amd64.deb && \
    apt-key add /var/nv-tensorrt-repo-ubuntu2004-cuda11.3-trt8.0.3.4-ga-20210831/7fa2af80.pub && \
    apt-get update && \
    apt-get install -y tensorrt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -f nv-tensorrt-repo-ubuntu2004-cuda11.3-trt8.0.3.4-ga-20210831_1-1_amd64.deb



WORKDIR /opt/tritonserver
RUN rm -rf /opt/tritonserver/*
COPY --chown=1000:1000 --from=full /opt/tritonserver/nvidia_entrypoint.sh .
ENTRYPOINT ["/opt/tritonserver/nvidia_entrypoint.sh"]

ENV NVIDIA_BUILD_ID 28453983
LABEL com.nvidia.build.id=28453983
LABEL com.nvidia.build.ref=a8c3497c460014286e5293d32fcd8df9c99621c7

WORKDIR /opt/tritonserver
COPY --chown=1000:1000 --from=full /opt/tritonserver/LICENSE .
COPY --chown=1000:1000 --from=full /opt/tritonserver/TRITON_VERSION .
COPY --chown=1000:1000 --from=full /opt/tritonserver/NVIDIA_Deep_Learning_Container_License.pdf .
COPY --chown=1000:1000 --from=full /opt/tritonserver/bin bin/
COPY --chown=1000:1000 --from=full /opt/tritonserver/lib lib/
COPY --chown=1000:1000 --from=full /opt/tritonserver/include include/
# Copying over backends 
# Copying over repoagents 

LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true
COPY --chown=1000:1000 --from=full /usr/bin/serve /usr/bin/.

# Copying over oneflow
COPY ./liboneflow_cpp/lib/ /usr/local/nvidia/lib/
COPY ./libtriton_oneflow.so /opt/tritonserver/backends/oneflow/
