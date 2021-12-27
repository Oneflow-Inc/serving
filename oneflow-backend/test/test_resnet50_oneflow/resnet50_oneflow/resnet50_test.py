import numpy as np
import oneflow as flow
import oneflow.nn as nn
import argparse
import tritonclient.http as httpclient
from flowvision.models.resnet import resnet50


class MyGraph(nn.Graph):
  def __init__(self, model):
      super().__init__()
      self.model = model

  def build(self, *input):
      return self.model(*input)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--model',
                        required=True,
                        help="the model to request")
    FLAGS = parser.parse_args()

    triton_client = httpclient.InferenceServerClient(url='127.0.0.1:8000')

    image = np.random.randn(1, 3, 224, 224).astype(np.float32)
    model = resnet50(pretrained=True, progress=True)
    # model.eval()
    graph = MyGraph(model)
    flow_output = graph(flow.tensor(image)).numpy()

    inputs = []
    inputs.append(httpclient.InferInput('INPUT_0', image.shape, "FP32"))
    inputs[0].set_data_from_numpy(image, binary_data=True)
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('OUTPUT_0', binary_data=True))
    results = triton_client.infer(FLAGS.model, inputs=inputs, outputs=outputs)
    output_data0 = results.as_numpy('OUTPUT_0')
    print(flow_output[0][0:10])
    print(output_data0[0][0:10])
    assert np.allclose(flow_output, output_data0, rtol=1e-03, atol=1e-03)
