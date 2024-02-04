# Build From Source

To build from source, you need to build liboneflow first.

1. Build liboneflow from source

    ```
    git clone https://github.com/Oneflow-Inc/oneflow --depth=1
    cd oneflow
    mkdir build && cd build
    cmake -C ../cmake/caches/cn/cuda.cmake -DBUILD_CPP_API=ON -DWITH_MLIR=ON -G Ninja ..
    ninja
    ```


2. Build oneflow backend from source

    ```
    mkdir build && cd build
    cmake -DCMAKE_PREFIX_PATH=/path/to/liboneflow_cpp/share -DTRITON_RELATED_REPO_TAG=r21.10 \
      -DTRITON_ENABLE_GPU=ON -G Ninja -DTHIRD_PARTY_MIRROR=aliyun ..
    ninja
    ```


3. Launch triton server

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


If you want to use XLA, TensorRT and OpenVINO in OneFlow-Serving, please build OneFlow-XRT and rebuild oneflow backend.

4. Build OneFlow-XRT with XLA, TensorRT or OpenVINO

   ```shell
   git clone https://github.com/Oneflow-Inc/oneflow-xrt.git
   cd oneflow-xrt
   mkdir build && cd build
   
   # Build OneFlow-XRT XLA
   cmake -G Ninja .. -DBUILD_XLA=ON && ninja
   
   # Build OneFlow-XRT TensorRT
   cmake -G Ninja .. -DBUILD_TENSORRT=ON -DTENSORRT_ROOT=/path/to/tensorrt && ninja
   
   # Build OneFlow-XRT OpenVINO
   cmake -G Ninja .. -DBUILD_OPENVINO=ON -DOPENVINO_ROOT=/path/to/openvino && ninja
   ```

5. Build oneflow backend from source

   ```shell
   mkdir build && cd build

   # Use TensorRT
   cmake -DCMAKE_PREFIX_PATH=/path/to/liboneflow_cpp/share -DTRITON_RELATED_REPO_TAG=r21.10 \
     -DTRITON_ENABLE_GPU=ON -DUSE_TENSORRT=ON -DONEFLOW_XRT_ROOT=$(pwd)/oneflow-xrt/build/install -G Ninja -DTHIRD_PARTY_MIRROR=aliyun ..
   ninja

   # Use XLA
   cmake -DCMAKE_PREFIX_PATH=/path/to/liboneflow_cpp/share -DTRITON_RELATED_REPO_TAG=r21.10 \
     -DTRITON_ENABLE_GPU=ON -DUSE_XLA=ON -DONEFLOW_XRT_ROOT=$(pwd)/oneflow-xrt/build/install -G Ninja -DTHIRD_PARTY_MIRROR=aliyun ..
   ninja

   # Use OpenVINO
   cmake -DCMAKE_PREFIX_PATH=/path/to/liboneflow_cpp/share -DTRITON_RELATED_REPO_TAG=r21.10 \
     -DTRITON_ENABLE_GPU=ON -DUSE_OPENVINO=ON -DONEFLOW_XRT_ROOT=$(pwd)/oneflow-xrt/build/install -G Ninja -DTHIRD_PARTY_MIRROR=aliyun ..
   ninja
   ```

6. Launch triton server

   ```shell
   cd ../  # back to root of the serving
   docker run --runtime=nvidia --rm --network=host \
     -v$(pwd)/examples:/models \
     -v$(pwd)/build/libtriton_oneflow.so:/backends/oneflow/libtriton_oneflow.so \
     -v$(pwd)/oneflow/build/liboneflow_cpp/lib/:/mylib \
     -v$(pwd)/oneflow-xrt/build/install/lib:/xrt_libs \
     nvcr.io/nvidia/tritonserver:21.10-py3 \
     bash -c 'LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mylib/:/xrt_libs /opt/tritonserver/bin/tritonserver \
     --model-repository=/models --backend-directory=/backends' 
   curl -v localhost:8000/v2/health/ready  # ready check
   ```
