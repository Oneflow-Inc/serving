import oneflow as flow
import oneflow.nn as nn
from flowvision.models.resnet import resnet50


if __name__ == "__main__":
    model = resnet50(pretrained=True, progress=True)
    model.eval()
    flow.save(model.state_dict(), "1/model")
