# Command Line Tool: oneflow-serving

We provide a command line tool `oneflow-serving` to help you to configure the model, if you do not want to write the model configuration file, you can use the command line arguments to configure the model. Note that only one of the two options is supported, configuration file or command line arguments. If the configuration file `config.pbtxt` exists, then the command line arguments will be ignored.

Example:

```
docker run -it --rm -v$(pwd)/triton-models:/models --runtime=nvidia --network=host \
  oneflowinc/oneflow-serving oneflow-serving --model-store /models --enable-tensorrt resnet101
```

The resnet101 is the name of a model directory under the model repository directory `/models`. Using the above command, resnet101 will enable xrt tensorrt backend to accelerate computation. 

## Arguments

The following command line arguments are briefly described as supported.

`--enable-openvino model_name`: specify the model name that wants to enable openvino
`--enable-tensorrt model_name`: specify the model name that wants to enable tensorrt
