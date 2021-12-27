import oneflow as flow
import oneflow.nn as nn
from flowvision.models.resnet import resnet50


class MyGraph(nn.Graph):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def build(self, *input):
        return self.model(*input)


if __name__ == "__main__":
    image = flow.ones((1, 3, 224, 224))
    model = resnet50(pretrained=True, progress=True)
    # model.eval()
    graph = MyGraph(model)
    out = graph(image)
    flow.save(graph, "1/model")
