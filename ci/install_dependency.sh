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

pip3 install flowvision
pip3 install tritonclient[all]

# build oneflow-backend
cd /ofserving
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install  -DTRITON_BACKEND_REPO_TAG=r21.10 -DTRITON_CORE_REPO_TAG=r21.10 -DTRITON_COMMON_REPO_TAG=r21.10 -G Ninja -DCMAKE_PREFIX_PATH=/liboneflow/share -DTRITON_ENABLE_GPU=ON ..
ninja
mkdir /opt/tritonserver/backends
mkdir /opt/tritonserver/backends/oneflow
cp libtriton_oneflow.so /opt/tritonserver/backends/oneflow
