import numpy as np
import pystk
import torch
from PIL import Image
import torchvision
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
#import torchvision.transforms.functional as TF
#from . import dense_transforms

IMAGE_PUCK_PATH = 'data/images'
PUCK_PATH = 'data/puck'

START = 0
END = 7000
LIMIT_COUNT = 0

class PuckLocationDataset(Dataset):
    def __init__(self):
        global LIMIT_COUNT, START, END
        from glob import glob
        import os
        self.images = []
        self.puck = []
        resize = torchvision.transforms.Resize([150, 200])

        print('loading images...')
        for file in os.listdir(IMAGE_PUCK_PATH):
            if LIMIT_COUNT < START:
                LIMIT_COUNT += 1
                continue
            if LIMIT_COUNT == END:
                break
            I = Image.open(os.path.join(IMAGE_PUCK_PATH,file))
            I = resize(I)
            I = F.to_tensor(I)
            self.images.append(I)
            LIMIT_COUNT += 1
            print('\rimage count: %d' % LIMIT_COUNT, end='\r')

        LIMIT_COUNT = 0

        print('loading puck peaks...')
        for file in os.listdir(PUCK_PATH):
            if LIMIT_COUNT < START:
                LIMIT_COUNT += 1
                continue
            if LIMIT_COUNT == END:
                break
            I = Image.open(os.path.join(PUCK_PATH,file))
            I = F.to_tensor(I)
            peak = self.extract_peak(I)
            self.puck.append(peak)
            LIMIT_COUNT += 1
            print('\rpuck count: %d' % LIMIT_COUNT, end='\r')

    def extract_peak(self, image):
        nz = torch.nonzero(image)

        if nz.numel() == 0:
            # predict a location behind the player
            ret_ten = torch.Tensor([200, 260])
            return ret_ten

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

class PuckVecDataset(Dataset):
    def __init__(self):
        import pickle
        print('loading inputs...')
        self.inputs = pickle.load(open("data/puck_info.p", "rb"))
        print('loading puck vecs...')
        self.puck_vecs = pickle.load(open("data/puck_vec.p", "rb"))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        one_input = self.inputs[idx]
        puck_vec = self.puck_vecs[idx]
        return one_input, puck_vec

def load_loc_data(num_workers=4, batch_size=32):
    dataset = PuckLocationDataset()
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

def load_vec_data(num_workers=4, batch_size=32):
    dataset = PuckVecDataset()
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

if __name__ == '__main__':
    load_loc_data()
