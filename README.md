# OneFlow Serving

[![Docker Image Version](https://img.shields.io/docker/v/oneflowinc/oneflow-serving?sort=semver)](https://hub.docker.com/r/oneflowinc/oneflow-serving)
[![Docker Pulls](https://img.shields.io/docker/pulls/oneflowinc/oneflow-serving)](https://hub.docker.com/r/oneflowinc/oneflow-serving)
[![License](https://img.shields.io/github/license/oneflow-inc/serving)](https://github.com/Oneflow-Inc/serving/blob/main/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Oneflow-Inc/serving/pulls)

Currently, we support [oneflow-backend](./oneflow-backend) for the [Triton Inference Server](https://github.com/triton-inference-server/server) that enables model serving.

# Triton Inference Server OneFlow Backend

OneFlow Backend For Triton Inference Server

## Get Started

Here is a [tutorial](./doc/tutorial.md) about how to export the model and how to deploy it. You can also follow the instructions below to get started.

1. Download and save model

  ```
  cd examples/resnet50/
  python3 export_model.py
  ```

2. Launch triton server

  ```
  cd ../../  # back to root of the serving
  docker run --rm --runtime=nvidia --network=host -v$(pwd)/oneflow-backend/examples:/models oneflowinc/oneflow-serving
  curl -v localhost:8000/v2/health/ready  # ready check
  ```

3. Send images and predict

  ```
  pip3 install tritonclient[all]
  cd examples/resnet50/
  curl -o cat.jpg https://images.pexels.com/photos/156934/pexels-photo-156934.jpeg
  python3 client.py --image cat.jpg
  ```

# Documentation

- [Tutorial](./doc/tutorial.md)
- [Build](./doc/build.md)
- [Model Configuration](./doc/model_config.md)
- [Know Issues](./doc/know_issues.md)
