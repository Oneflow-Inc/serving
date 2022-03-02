import os
import shutil
from time import time
import oneflow as flow
import oneflow.nn as nn
import flowvision


MODEL_ROOT = os.path.join(os.getcwd(), "repos")
LOG_ROOT = os.path.join(os.getcwd(), "log")
CONFIG_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.pbtxt")
DEVICE = "cuda:0"


class MyGraph(nn.Graph):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def build(self, *input):
        return self.model(*input)


def export_models(model_names, image):
    for model_name in model_names:        
        try:
            model = getattr(flowvision.models, model_name)()
            model.to(DEVICE)
            model.eval()
            graph = MyGraph(model)

            # start ticking
            out = model(image)
            s = out.shape
            t0 = time()
            out = model(image)
            s = out.shape
            t1 = time()
            out = graph(image)
            s = out.shape
            t2 = time()
            out = graph(image)
            s = out.shape
            t3 = time()
            print(model_name, "Model forward time: ", t1 - t0,
                              "Graph compile time: ", t2 - t1,
                              "Graph forward time: ", t3 - t2)

            model_dir = os.path.join(MODEL_ROOT, model_name + "_repo", model_name, "1", "model")
            config_dest = os.path.join(MODEL_ROOT, model_name + "_repo", model_name, "config.pbtxt")
            os.makedirs(model_dir, exist_ok=True)
            shutil.copyfile(CONFIG_FILE, config_dest)
            flow.save(graph, model_dir)
        except Exception as e:
            print(model_name, " cannot forward or convert to graph: ", e)


if __name__ == "__main__":
    model_names = ["alexnet", "efficientnet_b7", "mobilenet_v3_large", "resnet50", 
                  "resnet101", "vgg19", "vit_base_patch16_224", "mlp_mixer_b16_224"]
    image = flow.randn(1, 3, 224, 224).to(DEVICE)
    export_models(model_names, image)
