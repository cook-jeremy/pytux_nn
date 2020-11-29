import numpy as np
import pystk
import torch
from PIL import Image
import torchvision
from torchvision.transforms import functional as F

from torch.utils.data import Dataset, DataLoader
#import torchvision.transforms.functional as TF
#from . import dense_transforms

IMAGE_PATH = 'data/images'
PUCK_PATH = 'data/puck'


class SuperTuxDataset(Dataset):
    def __init__(self):
        from glob import glob
        import os
        self.images = []
        self.puck = []
        resize = torchvision.transforms.Resize([128, 128])

        for file in os.listdir(IMAGE_PATH):
            I = Image.open(os.path.join(IMAGE_PATH,file))
            I = resize(I)
            I = F.to_tensor(I)
            self.images.append(I)

        for file in os.listdir(PUCK_PATH):
            I = Image.open(os.path.join(PUCK_PATH,file))
            I = F.to_tensor(I)
            peak = extract_peak(I)
            self.puck.append(peak)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = self.transform(*image)
        puck = self.puck[idx]
        return image, puck

def extract_peak(image):
    nz = torch.nonzero(image)

    if nz.numel() == 0:
        return None

    xs = []
    ys = []
    for i in enumerate(nz):
        xs.append(i[1][1].item())
        ys.append(i[1][2].item())
    
    avg_x = sum(xs) / len(xs)
    avg_y = sum(ys) / len(ys)
    return (avg_x, avg_y)


def load_data(num_workers=0, batch_size=128):
    dataset = SuperTuxDataset()
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


if __name__ == '__main__':
    load_data()
