import os
import torch

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image

from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

from utils import construct_dataframe,visualize_data


class DatasetModule(Dataset):
    def __init__(self, transform=None, target_transform=None):
        
        self.df = construct_dataframe()
        self.bounding_boxes = self.df['labels']
        
        self.img_paths = self.df['img_path']
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        image_path = self.df.loc[idx,['img_path']].item() # item['img_path'] 
        label = self.df.loc[idx,['labels']].item()
        
        image = Image.open(image_path)
        image = torch.tensor(np.array(image))
        
        if self.transform:
            image = self.transform(image)
            
           #this doesnt work so I need ot see how to move the bb with the transformation , use albumentation   
        if self.target_transform:
            label = self.target_transform(label)
        return image, label