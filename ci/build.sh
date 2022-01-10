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

# build oneflow
cd oneflow
mkdir build
cd build
cmake .. -C ../cmake/caches/cn/cuda.cmake -DBUILD_CPP_API=ON -DBUILD_SHARED_LIBS=ON -DWITH_MLIR=ON -G Ninja
ninja
export ONEFLOW_BUILD=$(pwd)
export PYTHONPATH=$(pwd)/../python

# build oneflow-backend
export TRITON_VER=r21.10
cd ../../oneflow-backend
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install  -DTRITON_BACKEND_REPO_TAG=$TRITON_VER -DTRITON_CORE_REPO_TAG=$TRITON_VER -DTRITON_COMMON_REPO_TAG=$TRITON_VER -G Ninja -DCMAKE_PREFIX_PATH=$ONEFLOW_BUILD/liboneflow_cpp/share -DTRITON_ENABLE_GPU=ON ..
ninja

# install flowvision, run export model
pip3 install flowvision
cd ../../ci/test_resnet50_oneflow/resnet50_oneflow
python3 export_model.py
