import numpy as np
import oneflow as flow
import oneflow.nn as nn
import tritonclient.http as httpclient
from PIL import Image
from flowvision.models.resnet import resnet50


class MyGraph(nn.Graph):
  def __init__(self, model):
      super().__init__()
      self.model = model
  
  def build(self, *input):
      return self.model(*input)


if __name__ == '__main__':
    triton_client = httpclient.InferenceServerClient(url='127.0.0.1:8000')
    image = Image.open('./cat.jpg')
    
    image = image.resize((224, 224), Image.ANTIALIAS)
    image = np.asarray(image)
    image = image / 255
    image = np.expand_dims(image, axis=0)
    image = np.transpose(image, axes=[0, 3, 1, 2])
    image = image.astype(np.float32)

    model = resnet50(pretrained=True, progress=True)
    graph = MyGraph(model)
    flow_output = graph(flow.tensor(image)).numpy()

    inputs = []
    inputs.append(httpclient.InferInput('INPUT_0', image.shape, "FP32"))
    inputs[0].set_data_from_numpy(image, binary_data=False)
    outputs = []
    # outputs.append(httpclient.InferRequestedOutput('OUTPUT_0', binary_data=False, class_count=3))
    outputs.append(httpclient.InferRequestedOutput('OUTPUT_0', binary_data=False))

    results = triton_client.infer('resnet50_oneflow', inputs=inputs, outputs=outputs)
    output_data0 = results.as_numpy('OUTPUT_0')

    assert np.allclose(flow_output, output_data0, rtol=1e-05, atol=1e-05)
