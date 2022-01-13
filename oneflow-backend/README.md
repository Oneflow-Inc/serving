# Triton Inference Server OneFlow Backend

OneFlow Backend For Triton Inference Server

## Build

```
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install  -DTRITON_BACKEND_REPO_TAG=r21.10 -DTRITON_CORE_REPO_TAG=r21.10 -DTRITON_COMMON_REPO_TAG=r21.10 -G Ninja -DCMAKE_PREFIX_PATH=/triton/oneflow/build-clang/liboneflow/share -DTRITON_ENABLE_GPU=ON ..
ninja
```

## Model Repository

The Model Examples are located here: `docs/examples/model_repository`

## Run

```
nvidia-docker run --rm --runtime=nvidia --shm-size=2g --network=host -it --name triton-server -v `pwd`:/triton nvcr.io/nvidia/tritonserver:21.10-py3 bash
apt update && apt install libopenblas-dev
export LD_LIBRARY_PATH=/triton/  # /triton has liboneflow.so
./bin/tritonserver --model-store ./models  # put your models in ./models
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
