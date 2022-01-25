#!/usr/bin/env bash
set -euxo pipefail

# build oneflow-backend
git config --global http.proxy ${HTTP_PROXY}
git config --global https.proxy ${HTTP_PROXY}
cd oneflow-backend
mkdir -p build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install  -DTRITON_BACKEND_REPO_TAG=r$TRITON_VERSION -DTRITON_CORE_REPO_TAG=r$TRITON_VERSION -DTRITON_COMMON_REPO_TAG=r$TRITON_VERSION -G Ninja -DCMAKE_PREFIX_PATH=$ONEFLOW_CI_BUILD_DIR/liboneflow_cpp/share -DTRITON_ENABLE_GPU=ON ..
ninja -j16
