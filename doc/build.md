# Build From Source

To build from source, you need to build liboneflow first.

Build liboneflow from source

```
git clone https://github.com/Oneflow-Inc/oneflow --depth=1
cd oneflow
mkdir build
cd build
cmake .. -C ../cmake/caches/cn/cuda.cmake -DBUILD_CPP_API=ON -DBUILD_SHARED_LIBS=ON \
  -DWITH_MLIR=ON -G Ninja
ninja
```

Build oneflow backend from source

```
export TRITON_VER=r21.10
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install  -DTRITON_BACKEND_REPO_TAG=${TRITON_VER} \
  -DTRITON_CORE_REPO_TAG=${TRITON_VER} -DTRITON_COMMON_REPO_TAG=${TRITON_VER} -G Ninja \
  -DCMAKE_PREFIX_PATH=/path/to/liboneflow_cpp -DTRITON_ENABLE_GPU=ON ..
ninja
```

Launch triton server

```
cd ../../  # back to root of the serving
docker run --runtime=nvidia --rm --network=host \
  -v$(pwd)/oneflow-backend/examples:/models \
  -v$(pwd)/oneflow-backend/build/libtriton_oneflow.so:/backends/oneflow/libtriton_oneflow.so \
  -v$(pwd)/oneflow/build/liboneflow_cpp/lib/:/mylib nvcr.io/nvidia/tritonserver:21.10-py3 \
  bash -c 'LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mylib/ /opt/tritonserver/bin/tritonserver \
  --model-repository=/models --backend-directory=/backends' 
curl -v localhost:8000/v2/health/ready  # ready check
```
