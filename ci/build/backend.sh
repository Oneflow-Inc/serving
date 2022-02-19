#!/usr/bin/env bash
set -euxo pipefail

# build oneflow-backend
git config --global http.proxy ${HTTP_PROXY}
git config --global https.proxy ${HTTP_PROXY}

mkdir -p build
cd build
cmake -DCMAKE_PREFIX_PATH=$ONEFLOW_CI_BUILD_DIR/liboneflow_cpp/share \
    -DTRITON_RELATED_REPO_TAG=r$TRITON_VERSION \
    -DTRITON_ENABLE_GPU=ON \
    -DTHIRD_PARTY_MIRROR=aliyun \
    -G Ninja ..
ninja -j8
