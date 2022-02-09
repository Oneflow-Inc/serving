# OneFlow Serving

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Oneflow-Inc/serving/pulls)

Currently, we support [oneflow-backend](./oneflow-backend) for the [Triton Inference Server](https://github.com/triton-inference-server/server) that enables model serving.

# Triton Inference Server OneFlow Backend

OneFlow Backend For Triton Inference Server

## Get Started

Download and save model

```
cd examples/resnet50_oneflow/
python3 export_model.py
```

Launch triton server

```
cd ../../  # back to root of the serving
docker run --rm --runtime=nvidia --network=host -v$(pwd)/oneflow-backend/examples:/models 
oneflowinc/oneflow-serving:0.0.1 /opt/tritonserver/bin/tritonserver --model-store /models
curl -v localhost:8000/v2/health/ready  # ready check
```

Send images and predict

```
pip3 install tritonclient[all]
cd examples/resnet50_oneflow/
python3 client.py --image cat.jpg
```

## Build From Source

To build from source, you need to build liboneflow first.

Build liboneflow from source

```
git clone https://github.com/Oneflow-Inc/oneflow --depth=1
cd oneflow
mkdir build
cd build
cmake .. -C ../cmake/caches/cn/cuda.cmake -DBUILD_CPP_API=ON -DBUILD_SHARED_LIBS=ON 
-DWITH_MLIR=ON -G Ninja
ninja
```

Build oneflow backend from source

```
export TRITON_VER=r21.10
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install  -DTRITON_BACKEND_REPO_TAG=${TRITON_VER}
-DTRITON_CORE_REPO_TAG=${TRITON_VER} -DTRITON_COMMON_REPO_TAG=${TRITON_VER} -G Ninja
-DCMAKE_PREFIX_PATH=/path/to/liboneflow_cpp -DTRITON_ENABLE_GPU=ON ..
ninja
```

Launch triton server

```
cd ../../  # back to root of the serving
docker run --runtime=nvidia --rm -p8000:8000 -p8001:8001 -p8002:8002 
-v$(pwd)/oneflow-backend/examples:/models 
-v$(pwd)/oneflow-backend/build/libtriton_oneflow.so:/backends/oneflow/libtriton_oneflow.so 
-v$(pwd)/oneflow/build/liboneflow_cpp/lib/:/mylib nvcr.io/nvidia/tritonserver:21.10-py3 
bash -c 'LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mylib/ /opt/tritonserver/bin/tritonserver 
--model-repository=/models --backend-directory=/backends' 
curl -v localhost:8000/v2/health/ready  # ready check
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

## Known Issues

### llvm: Option already exists

Oneflow backend conflits with tensorflow1 due to some mysterious reason. It is recommended not to use oneflow and tensorflow1 together.

```
.../llvm/include/llvm/Support/CommandLine.h:858: void llvm::cl::parser<DataType>::addLiteralOption
(llvm::StringRef, const DT&, llvm::StringRef) [with DT = llvm::FunctionPass* (*)(); DataType = 
llvm::FunctionPass* (*)()]: Assertion `findOption(Name) == Values.size() && "Option already 
exists!"' failed.
```
