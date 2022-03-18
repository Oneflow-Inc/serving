# OneFlow Serving

[![Docker Image Version](https://img.shields.io/docker/v/oneflowinc/oneflow-serving?sort=semver)](https://hub.docker.com/r/oneflowinc/oneflow-serving)
[![Docker Pulls](https://img.shields.io/docker/pulls/oneflowinc/oneflow-serving)](https://hub.docker.com/r/oneflowinc/oneflow-serving)
[![License](https://img.shields.io/github/license/oneflow-inc/serving)](https://github.com/Oneflow-Inc/serving/blob/main/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Oneflow-Inc/serving/pulls)

Currently, we support [oneflow-backend](./oneflow-backend) for the [Triton Inference Server](https://github.com/triton-inference-server/server) that enables model serving.

# Triton Inference Server OneFlow Backend

OneFlow Backend For Triton Inference Server

## Get Started

Download and save model

```
cd examples/resnet50/
python3 export_model.py
```

Launch triton server

```
cd ../../  # back to root of the serving
docker run --rm --runtime=nvidia --network=host -v$(pwd)/oneflow-backend/examples:/models oneflowinc/oneflow-serving:0.0.1
curl -v localhost:8000/v2/health/ready  # ready check
```

Send images and predict

```
pip3 install tritonclient[all]
cd examples/resnet50/
curl -o cat.jpg https://images.pexels.com/photos/156934/pexels-photo-156934.jpeg
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
    string_value: "tensorrt"
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
