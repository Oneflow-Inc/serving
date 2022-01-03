<!--
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Triton Inference Server OneFlow Backend

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Oneflow-Inc/serving/pulls)

OneFlow Backend For Triton Inference Server

## Quick Start

Pull Docker image

```
docker pull oneflow-serving:0.1
```

Download and save model

```
cd examples/resnet50_oneflow/
python3 model.py
docker run --runtime=nvidia --rm  -v$(pwd):$(pwd) -w $(pwd) oneflow-serving:0.1 python3 model.py
cd ../..
```

Launch triton server

```
docker run --runtime=nvidia --rm -p8000:8000 -p8001:8001 -p8002:8002 -v$(pwd)/examples:/models oneflow-serving:0.1 /opt/tritonserver/bin/tritonserver --model-repository=/models
curl -v localhost:8000/v2/health/ready  # ready check
```

Send images and predict

```
cd examples/resnet50_oneflow/
pip3 install tritonclient[all]
python3 client.py --image images/cat.jpg
python3 client.py --image images/dog.jpg
```

## Build

build oneflow backend from source

```
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install  -DTRITON_BACKEND_REPO_TAG=r21.10 -DTRITON_CORE_REPO_TAG=r21.10 -DTRITON_COMMON_REPO_TAG=r21.10 -G Ninja -DCMAKE_PREFIX_PATH=/path/to/liboneflow_cpp -DTRITON_ENABLE_GPU=ON ..
ninja
```

## Model Config Convention

### input output name

- input name should follow naming convention: `INPUT_<index>`
- output name should follow naming convention: `OUTPUT_<index>`
- In `INPUT_<index>`, the `<index>` should be in the range `[0, input_size)`
- In `OUTPUT_<index>`, the `<index>` should be in the range `[0, output_size)`

### XRT

You can enable XRT by adding following configuration.

```
parameters {
  key: "xrt"
  value: {
    string_value: "openvino"
  }
}
```

### Model Repository Structure

A directory named `model` should be put in the version directory.

Example:

```
.
├── 1
│   └── model
├── client.py
├── config.pbtxt
├── labels.txt
└── model.py
```

#### Model Backend Name

Model backend name must be `oneflow`.

```
name: "identity"
backend: "oneflow"
```
