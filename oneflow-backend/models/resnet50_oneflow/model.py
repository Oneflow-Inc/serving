import oneflow as flow
import oneflow.nn as nn
from flowvision.models.resnet import resnet50
import time


class MyGraph(nn.Graph):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def build(self, *input):
        return self.model(*input)


if __name__ == "__main__":
    model = resnet50(pretrained=True, progress=True)
    image = flow.ones((1, 3, 224, 224))
    model = model.to("cuda:0")
    image = image.to("cuda:0")
    y = model(image)

    now = time.time()
    y = model(image)
    print(y.shape)
    print(time.time() - now)

    now = time.time()
    graph = MyGraph(model)
    y = graph(image)
    print("compile time: ", time.time() - now)

    now = time.time()
    y = graph(image)
    y.numpy()
    print(y.shape)
    print(time.time() - now)

    flow.save(graph, "1/model")
