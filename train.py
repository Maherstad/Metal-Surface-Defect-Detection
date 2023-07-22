import comet_ml
import torch
from ultralytics import YOLO


 # load a pretrained model (recommended for training) #'yolov8l.pt'
#model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
#http://karpathy.github.io/2019/04/25/recipe/

if __name__ == '__main__':
    # Train the model
    
    model = YOLO('./project1/exp16/weights/best.pt') 
    model.train(resume=True)
    # model.train(data='./yolov5/custom_dataset.yaml',
    #             epochs=10,
    #             batch = 8,
    #             imgsz=640,
    #             patience=20,
    #             device='cuda', #gpu
    #             project= 'project1', #project name 
    #             name = 'exp1', #experiment name
    #             resume = True
               
    #            )
