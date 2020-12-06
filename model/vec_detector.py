import torch
import torch.nn.functional as F

class VecDetector(torch.nn.Module):

    def __init__(self, input_dim=6, output_dim=2):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.l1 = torch.nn.Linear(input_dim, 512)
        self.l2 = torch.nn.Linear(512, 512)
        self.l3 = torch.nn.Linear(512, 512)
        self.classifier = torch.nn.Linear(512, output_dim)


    def forward(self, x):
        z = self.relu(self.l1(x))
        z = self.relu(self.l2(z))
        z = self.relu(self.l3(z))
        return self.classifier(z)

def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, VecDetector):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'puck_vec.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))

def load_model():
    from torch import load
    from os import path
    r = VecDetector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'puck_vec.th'), map_location='cpu'))
    return r