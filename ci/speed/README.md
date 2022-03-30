# Speed Test

为了运行测试，需要在项目根目录运行如下命令。

1. 导出模型

```
MODEL_NAMES="alexnet efficientnet_b7 mobilenet_v3_large resnet50 resnet101 \
    vgg19 vit_base_patch16_224 mlp_mixer_b16_224"

docker pull oneflowinc/oneflow:nightly-cuda11.2
docker run --rm -v $(pwd):$(pwd) -w $(pwd) --runtime=nvidia --network=host --detach \
    --name oneflow-mlir-container --shm-size=8g oneflowinc/oneflow:nightly-cuda11.2 sleep 7200
docker exec oneflow-mlir-container python3 -m pip install --upgrade --force-reinstall \
    --find-links=https://oneflow-staging.oss-cn-beijing.aliyuncs.com/canary/refs/heads/master/cu112/index.html oneflow
docker exec oneflow-mlir-container python3 -m pip install flowvision \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
docker exec --env MODEL_NAMES="$MODEL_NAMES" oneflow-mlir-container \
    python3 ci/speed/models.py "$MODEL_NAMES"
```

2. 启动测试脚本

之后会产生一个 `speed_test_output` 文件夹，里面存放有详细的测试数据。其中 `speed_test_*` 开头的几个文件，分别存放有不同设备上的详细和摘要数据。

```
docker run --rm -v $PWD/repos:/p -w /p busybox chmod -R o+w .
python3 ci/speed/speed.py --model_names "$MODEL_NAMES" --device cuda:0 --xrt tensorrt
python3 ci/speed/speed.py --model_names "$MODEL_NAMES" --device cuda:0
python3 ci/speed/speed.py --model_names "$MODEL_NAMES" --xrt openvino
python3 ci/speed/speed.py --model_names "$MODEL_NAMES"
```

