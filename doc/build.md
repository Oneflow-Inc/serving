# Build From Source

You can pull the docker images, and follow the instructions below to build in docker container.

```
docker pull registry.cn-beijing.aliyuncs.com/oneflow/triton-devel
```

To build from source, you need to build liboneflow first.

- Build liboneflow from source

```
git clone https://github.com/Oneflow-Inc/oneflow --depth=1
cd oneflow
mkdir build && cd build
cmake -C ../cmake/caches/cn/cuda.cmake -DBUILD_CPP_API=ON -DBUILD_SHARED_LIBS=ON \
  -DWITH_MLIR=ON -G Ninja ..
ninja
```

- Build oneflow backend from source

```
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/liboneflow_cpp/share -DTRITON_RELATED_REPO_TAG=r21.10 \
  -DTRITON_ENABLE_GPU=ON -G Ninja -DTHIRD_PARTY_MIRROR=aliyun ..
ninja
```

- Launch triton server

```
cd ../  # back to root of the serving
docker run --runtime=nvidia --rm --network=host \
  -v$(pwd)/examples:/models \
  -v$(pwd)/build/libtriton_oneflow.so:/backends/oneflow/libtriton_oneflow.so \
  -v$(pwd)/oneflow/build/liboneflow_cpp/lib/:/mylib nvcr.io/nvidia/tritonserver:21.10-py3 \
  bash -c 'LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mylib/ /opt/tritonserver/bin/tritonserver \
  --model-repository=/models --backend-directory=/backends' 
curl -v localhost:8000/v2/health/ready  # ready check
```
