import torch
import torch.nn.functional as F

class PuckSeg(torch.nn.Module):
    def __init__(self, c_in=3, c_out=1):
        super().__init__()

        self.relu = torch.nn.ReLU()
        self.norm = torch.nn.BatchNorm2d(c_in)

        self.down1 = torch.nn.Conv2d(c_in, 32, kernel_size=3, stride=2, padding=1)
        self.down2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.down3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.up3 =  torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.up2 =  torch.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.up1 =  torch.nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)

        self.linear = torch.nn.Conv2d(16, c_out, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        """
        Translate the image to a mask of the horse.

        Input: 
            x (float tensor N x 3 x 128 x 128): input image of a horse
        Output:
            y (float tensor N x 1 x 128 x 128): binary mask of the horse
        """
        x = self.norm(x)
        down1 = self.relu(self.down1(x))
        down2 = self.relu(self.down2(down1))
        down3 = self.relu(self.down3(down2))
        up3 = self.relu(self.up3(down3))
        up2 = self.relu(self.up2(up3))
        up1 = self.relu(self.up1(up2))
        
        return self.linear(up1)

def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, PuckSeg):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'puck_seg.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))

def load_model():
    from torch import load
    from os import path
    r = PuckSeg()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'puck_seg.th'), map_location='cpu'))
    return r