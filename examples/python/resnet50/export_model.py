import oneflow as flow
import oneflow.nn as nn
from flowvision.models.resnet import resnet50


class MyGraph(nn.Graph):
    def __init__(self, model):
        super().__init__(enable_get_runtime_state_dict=True)
        self.model = model

    def build(self, *input):
        return self.model(*input)


if __name__ == "__main__":
    image = flow.ones((1, 3, 224, 224), device=flow.device("cuda:0"))
    model = resnet50(pretrained=True, progress=True).to(flow.device("cuda:0"))
    model.eval()
    graph = MyGraph(model)
    out = graph(image)
    flow.save(graph.runtime_state_dict(with_eager=True), "1/model")
