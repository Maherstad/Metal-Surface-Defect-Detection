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
    def __init__(self, data_df: pd.DataFrame, transform=None, target_transform=None):
        
        self.data_df = data_df
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        
        image_path = self.data_df.iloc[idx]['img_path']
        label = self.data_df.iloc[idx]['labels']
        
        image = Image.open(image_path)
        image = torch.tensor(np.array(image))
        
        if self.transform:
            image = self.transform(image)
            
           #this doesnt work so I need ot see how to move the bb with the transformation , use albumentation   
        if self.target_transform:
            label = self.target_transform(label)
        return image, label