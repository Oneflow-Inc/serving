# Copyright 2020 The OneFlow Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -xe

pip3 install flowvision

# TODO(zzk0): remove this
export HTTP_PROXY="http://192.168.1.11:8118"
git config --global http.proxy ${HTTP_PROXY}
git config --global https.proxy ${HTTP_PROXY}

# build oneflow
git clone https://github.com/Oneflow-Inc/oneflow --depth=1
cd oneflow
export PYTHONPATH=$(pwd)/python
mkdir build
cd build
cmake .. -C ../cmake/caches/cn/cuda.cmake -DBUILD_CPP_API=ON -DBUILD_MONOLITHIC_LIBONEFLOW_CPP_SO=ON -DBUILD_SHARED_LIBS=OFF -DWITH_MLIR=ON -G Ninja
ninja
export ONEFLOW_BUILD=$(pwd)
echo $ONEFLOW_BUILD

echo "------------------------------------------"
ls -al $ONEFLOW_BUILD/liboneflow_cpp
ls -al $ONEFLOW_BUILD/liboneflow_cpp/include/oneflow/
echo "------------------------------------------"

# build oneflow-backend
cd /ofserving
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install  -DTRITON_BACKEND_REPO_TAG=r21.10 -DTRITON_CORE_REPO_TAG=r21.10 -DTRITON_COMMON_REPO_TAG=r21.10 -G Ninja -DCMAKE_PREFIX_PATH=$ONEFLOW_BUILD/liboneflow_cpp/share -DTRITON_ENABLE_GPU=ON ..
ninja
mkdir /opt/tritonserver/backends
mkdir /opt/tritonserver/backends/oneflow
cp libtriton_oneflow.so /opt/tritonserver/backends/oneflow
