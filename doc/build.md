# Build From Source

To build from source, you need to build liboneflow first.

1. Build base image
    
    ```
    docker build -t serving:base . -f docker/Dockerfile.base
    ```

2. Build liboneflow in docker image
    
    ```
    docker build -t serving:build_of . -f docker/Dockerfile.build_of
    ```
3. Build backends in docker image
    
    ```
    docker build -t serving:final . -f docker/Dockerfile.serving
    ```

4. Launch triton server

    ```
    cd ../  # back to root of the serving
    docker run --runtime=nvidia \
      --rm \
      --network=host \
      -v$(pwd)/examples/cpp:/models \
      serving:final
    curl -v localhost:8000/v2/health/ready  # ready check
    ```
