set -euxo pipefail

# docker pull oneflowinc/oneflow:nightly-cuda11.2
docker run --rm -v $(pwd):$(pwd) -w $(pwd) --runtime=nvidia --network=host --detach --name oneflow-mlir-container --shm-size=8g oneflowinc/oneflow:nightly-cuda11.2 sleep 360000
docker exec oneflow-mlir-container python3 -m pip install --upgrade --force-reinstall  --find-links=https://oneflow-staging.oss-cn-beijing.aliyuncs.com/canary/refs/heads/master/cu112/index.html oneflow
docker exec oneflow-mlir-container python3 -m pip install flowvision
docker exec oneflow-mlir-container python3 ci/speed/models.py
python3 ci/speed/speed.py
python3 ci/speed/parse.py
docker stop oneflow-mlir-container
