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

set -uex

# TODO(zzk0): remove this
export HTTP_PROXY="http://192.168.1.12:10609"
git config --global http.proxy ${HTTP_PROXY}
git config --global https.proxy ${HTTP_PROXY}

# build oneflow
cd oneflow
mkdir -p build
cd build
cmake .. -C ../cmake/caches/cn/cuda.cmake -DBUILD_CPP_API=ON -DBUILD_SHARED_LIBS=ON -DWITH_MLIR=ON -G Ninja
ninja

# copy dependencies
cp oneflow/ir/lib/*.so liboneflow_cpp/lib/
cp oneflow/ir/lib/*.so.VERSION liboneflow_cpp/lib/
cp oneflow/ir/llvm_monorepo-build/lib/*.so liboneflow_cpp/lib/
cp oneflow/ir/llvm_monorepo-build/lib/*.so.14git liboneflow_cpp/lib/
cp third_party_install/glog/install/lib/libglog.so* liboneflow_cpp/lib/
cp third_party_install/protobuf/lib/libprotobuf.so* liboneflow_cpp/lib/
cp third_party_install/nccl/lib/libnccl.so* liboneflow_cpp/lib/
export ONEFLOW_BUILD=$(pwd)
export PYTHONPATH=$(pwd)/../python

# build oneflow-backend
cd ../../oneflow-backend
mkdir -p build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install  -DTRITON_BACKEND_REPO_TAG=r$TRITON_VERSION -DTRITON_CORE_REPO_TAG=r$TRITON_VERSION -DTRITON_COMMON_REPO_TAG=r$TRITON_VERSION -G Ninja -DCMAKE_PREFIX_PATH=$ONEFLOW_BUILD/liboneflow_cpp/share -DTRITON_ENABLE_GPU=ON ..
ninja

# install flowvision, run export model
cd ../../ci
pip3 install -r oneflow-requirement.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
cd ./test_resnet50_oneflow/resnet50_oneflow
python3 export_model.py