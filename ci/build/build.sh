#!/usr/bin/env bash
set -euxo pipefail

# build oneflow
cd oneflow
mkdir -p build
cd build
# cmake .. -C ../cmake/caches/cn/cuda.cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_CPP_API=ON -DBUILD_SHARED_LIBS=ON -DWITH_MLIR=ON -G Ninja
# ninja -j16

export ONEFLOW_BUILD=$(pwd)
export PYTHONPATH=$(pwd)/../python

# build oneflow-backend
git config --global http.proxy ${HTTP_PROXY}
git config --global https.proxy ${HTTP_PROXY}
cd ../../oneflow-backend
mkdir -p build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install  -DTRITON_BACKEND_REPO_TAG=r$TRITON_VERSION -DTRITON_CORE_REPO_TAG=r$TRITON_VERSION -DTRITON_COMMON_REPO_TAG=r$TRITON_VERSION -G Ninja -DCMAKE_PREFIX_PATH=$ONEFLOW_BUILD/liboneflow_cpp/share -DTRITON_ENABLE_GPU=ON ..
ninja -j16

# install flowvision, run export model
cd ../../ci
pip3 install -r build/requirement.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
cd ./test/test_resnet50_oneflow/resnet50_oneflow
python3 export_model.py
