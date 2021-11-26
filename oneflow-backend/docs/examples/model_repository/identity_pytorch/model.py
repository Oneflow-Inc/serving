import torch
import torch.nn as nn

class Relu(nn.Module):

  def __init__(self) -> None:
    super().__init__()
    self.relu = nn.ReLU()
  
  def forward(self, x):
    return self.relu(x)


model = Relu()
model.eval()
x = torch.randn(1, 10)
model_traced = torch.jit.trace(model, x)
model(x)
model_traced.save('model.pt')
