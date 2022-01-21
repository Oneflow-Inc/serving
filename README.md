# Oneflow Serving

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Oneflow-Inc/serving/pulls)

Currently, we support [oneflow-backend](./oneflow-backend) for the [Triton Inference Server](https://github.com/triton-inference-server/server) that enables model serving.

# Triton Inference Server OneFlow Backend

OneFlow Backend For Triton Inference Server

## Get Started

If you want to try it, you need to build liboneflow and oneflow-backend from source. An out-of-the-box docker image will be released soon.

Build liboneflow from source

```
git clone https://github.com/Oneflow-Inc/oneflow --depth=1
cd oneflow
mkdir build
cd build
cmake .. -C ../cmake/caches/cn/cuda.cmake -DBUILD_CPP_API=ON -DBUILD_SHARED_LIBS=ON -DWITH_MLIR=ON -G Ninja
ninja
```

Build oneflow backend from source

```
export TRITON_VER=r21.10
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/liboneflow_cpp/share -DTRITON_RELATED_REPO_TAG=r21.10 -DTRITON_ENABLE_GPU=ON -G Ninja ..
ninja
```

Download and save model

```
cd example/resnet50/
python3 export_model.py
```

Launch triton server

```
cd ../../  # back to root of the serving
docker run --runtime=nvidia --rm -p8000:8000 -p8001:8001 -p8002:8002 -v$(pwd)/oneflow-backend/examples:/models -v$(pwd)/oneflow-backend/build/libtriton_oneflow.so:/backends/oneflow/libtriton_oneflow.so -v$(pwd)/oneflow/build/liboneflow_cpp/lib/:/mylib nvcr.io/nvidia/tritonserver:21.10-py3 bash -c 'LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mylib/ /opt/tritonserver/bin/tritonserver --model-repository=/models --backend-directory=/backends' 
curl -v localhost:8000/v2/health/ready  # ready check
```

Send images and predict

```
pip3 install tritonclient[all]
cd examples/resnet50/
python3 client.py --image cat.jpg
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
.../llvm/include/llvm/Support/CommandLine.h:858: void llvm::cl::parser<DataType>::addLiteralOption(llvm::StringRef, const DT&, llvm::StringRef) [with DT = llvm::FunctionPass* (*)(); DataType = llvm::FunctionPass* (*)()]: Assertion `findOption(Name) == Values.size() && "Option already exists!"' failed.
```
