"""
# structure of the dataset needed is :
 * train:
   - labels  
   - images 
 * val:
   - images  
   - labels
   
   
with labels being text files that match the image name and consist of lines of bounding boxes



> transfer learning 
https://docs.ultralytics.com/yolov5/tutorials/transfer_learning_with_frozen_layers/  
> train with custom dataset 
https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/  
> youtube tutorial 
https://www.youtube.com/watch?v=GRtgLlwxpc4


"""

import opendatasets as od


import os
#import comet_ml

import torch
from sklearn.model_selection import train_test_split
from utils import construct_dataframe,transform_image_and_bbs,construct_yolo_compatible_data_structure


od.download('https://www.kaggle.com/datasets/zhangyunsheng/defects-class-and-location')

comet_ml.init()

paths = [
os.path.join(os.getcwd(),'yolo_dataset'),
os.path.join(os.getcwd(),'yolo_dataset','train'),
os.path.join(os.getcwd(),'yolo_dataset','train','images'),
os.path.join(os.getcwd(),'yolo_dataset','train','labels'),
os.path.join(os.getcwd(),'yolo_dataset','val'),
os.path.join(os.getcwd(),'yolo_dataset','val','images'),
os.path.join(os.getcwd(),'yolo_dataset','val', 'labels'),
]

for path in paths:
    if not os.path.exists(path):
        os.makedirs(path)

train_df, val_df = train_test_split(construct_dataframe(), shuffle = True,test_size=0.2, random_state=42)
construct_yolo_compatible_data_structure(train_df)
construct_yolo_compatible_data_structure(val_df)


