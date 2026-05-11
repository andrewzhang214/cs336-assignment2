import torch
import torch.nn as nn


class ToyModel(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        print(x.dtype)
        x = self.ln(x)
        print(x.dtype)
        x = self.fc2(x)
        print(x.dtype)
        return x

model : torch.nn.Module = ToyModel(5, 5) # e.g. your Transformer model
dtype : torch.dtype = torch.float16 # e.g. torch.float16
x : torch.Tensor = torch.rand(1, 5) # input data

with torch.autocast(device_type="cpu", dtype=dtype):
    print(next(model.parameters()).dtype)
    y = model(x)
    loss = y.mean()
    print(loss.dtype)
    loss.backward()

    for name, param in model.named_parameters():
        print(name)
        print(param.grad.dtype)