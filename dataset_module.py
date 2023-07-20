import os
import torch

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image

from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import datasets

import torchvision.transforms.v2 as transforms


from utils import construct_dataframe,visualize_data



class DatasetModule(Dataset):
    def __init__(self, data_df: pd.DataFrame, transform=False):
        
        self.data_df = data_df
        self.transform = transform

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        
        image_path = self.data_df.iloc[idx]['img_path']
        label = self.data_df.iloc[idx]['labels']
        
        image = Image.open(image_path)
        image = torch.tensor(np.array(image))
        
        if self.transform:
            
            trans = transforms.Compose([
                        transforms.RandomHorizontalFlip(0.9), #ideally around 0.5
                        transforms.RandomRotation(90),
                        transforms.Resize((360,360)),
                        transforms.ToTensor()])

            image,label = trans(image,label)

        return image, label