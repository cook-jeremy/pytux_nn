import numpy as np
import pystk
import torch
from PIL import Image
import torchvision
from torchvision.transforms import functional as F

from torch.utils.data import Dataset, DataLoader
#import torchvision.transforms.functional as TF
#from . import dense_transforms

IMAGE_PUCK_PATH = 'data/images/has_puck'
IMAGE_NO_PUCK_PATH = 'data/images/no_puck'
PUCK_PATH = 'data/puck'


class PuckLocationDataset(Dataset):
    def __init__(self):
        from glob import glob
        import os
        self.images = []
        self.puck = []
        resize = torchvision.transforms.Resize([150, 200])

        print('loading images...')
        for file in os.listdir(IMAGE_PUCK_PATH):
            I = Image.open(os.path.join(IMAGE_PUCK_PATH,file))
            I = resize(I)
            I = F.to_tensor(I)
            self.images.append(I)

        print('puck peaks...')
        for file in os.listdir(PUCK_PATH):
            I = Image.open(os.path.join(PUCK_PATH,file))
            I = F.to_tensor(I)
            peak = self.extract_peak(I)
            self.puck.append(peak)

    def extract_peak(self, image):
        nz = torch.nonzero(image)

        if nz.numel() == 0:
            print('ERROR IN LOADING PUCK PEAKS!!!')
            return [0,0]

        xs = []
        ys = []
        for i in enumerate(nz):
            xs.append(i[1][1].item())
            ys.append(i[1][2].item())
        
        avg_x = sum(xs) / len(xs)
        avg_y = sum(ys) / len(ys)
        ret_list = [avg_y, avg_x]
        ret_ten = torch.Tensor(ret_list)
        return ret_ten

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        puck = self.puck[idx]
        return image, puck

MAX_PUCK = 5000

class PuckIsDataset(Dataset):
    def __init__(self):
        from glob import glob
        import os
        self.images = []
        self.is_puck = []
        resize = torchvision.transforms.Resize([150, 200])

        image_count = 0
        for file in os.listdir(IMAGE_PUCK_PATH):
            # want an equal amount of images with puck and without puck, but data is skewed towards having puck so we need to limit
            if image_count == MAX_PUCK:
                break
            I = Image.open(os.path.join(IMAGE_PUCK_PATH,file))
            I = resize(I)
            I = F.to_tensor(I)
            self.images.append(I)
            self.is_puck.append(100.0)
            image_count += 1

        for file in os.listdir(IMAGE_NO_PUCK_PATH):
            I = Image.open(os.path.join(IMAGE_NO_PUCK_PATH,file))
            I = resize(I)
            I = F.to_tensor(I)
            self.images.append(I)
            self.is_puck.append(0.0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        puck = self.is_puck[idx]
        return image, puck


def load_loc_data(num_workers=4, batch_size=32):
    dataset = PuckLocationDataset()
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

def load_is_data(num_workers=4, batch_size=32):
    dataset = PuckIsDataset()
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

if __name__ == '__main__':
    load_loc_data()
