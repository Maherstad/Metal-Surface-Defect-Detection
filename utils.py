#imports 

import os
#import comet_ml

import torch
import cv2
import numpy as np
import pandas as pd 
from PIL import Image
import albumentations
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import matplotlib.patches as patches

import ultralytics
import opendatasets


defect_types_languages_list = { 
    '1_chongkong' : ('punching_hole','Stanzen'),'2_hanfeng' : ('welding_line','schweißbahn'),
    '3_yueyawan' : ('crescent_gap','sichelspalt'),'4_shuiban' : ('water_spot','wasserfleck'),
    '5_youban': ('oil_spot','ölfleck'),'6_siban': ('silk_spot','seidenfleck'),
    '7_yiwu' : ('inclusion','einschlüss'),'8_yahen' : ('rolled_pit','gewalzte_grube'),
    '9_zhehen' : ('crease','falten'),'10_yaozhed' : ('waist_folding','taillenfalten')
    }


"""
{'crease': 0,
 'silk_spot': 1,
 'water_spot': 2,
 'rolled_pit': 3,
 'inclusion': 4,
 'crescent_gap': 5,
 'welding_line': 6,
 'punching_hole': 7,
 'oil_spot': 8,
 'waist_folding': 9}

"""


def assign_language(word, word_to = 'eng'):
    # due to the dataset being in Chinese, we need to convert the defect types to English or German

    ret_value = None

    for key,value in defect_types_languages_list.items():
        if word in value:
            ret_value =  key
        elif word in key and word_to == 'ger':
            ret_value =  value[1]
        elif word in key and word_to == 'eng':
            ret_value =  value[0]

    return ret_value

def read_xml(xml_path: str) -> dict:
    """
    read the labeling files , xml in our case, and extract all useful information from the xml 
    
    input:
    
        xml_path : the path to the xml file , type == str 
        
    output: a dictionary consisting of the following key,value pairs 
    
    'dimensions'     : [width,height,depth]
    'bounding_boxes' : list of lists containing [[defect_categories[defect_type],xmin,ymin,xmax,ymax], ...] 
    'defect_types'   : list of str containing the defects detected in the image
    
     
    """
    
    tree_path = xml_path
    tree = ET.parse(tree_path)
    root = tree.getroot()

    filename = root.find('filename').text

    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)
    depth = int(root.find('size/depth').text)

    objects = root.findall('object')
    bbs = []
    defect_types=[]
    for obj in objects:
        
        name = obj.find('name').text
        defect_type = assign_language(name,'eng')
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)
        

#         incase one-hot-vector is needed
#         name = assign_language(name,'eng')
#         one_hot_vector = [0] * len(defect_categories)
#         defect_index = defect_categories[name]
#         one_hot_vector[defect_index] =1
        
        defect_categories = extract_unique_defect_types()
        bbs.append([defect_categories[defect_type],xmin,ymin,xmax,ymax])
        defect_types.append(defect_type)
        
        
    return { 'dimensions'     : [width,height,depth],
             'bounding_boxes' : bbs,
            'defect_types':defect_types
           }
    
#np.random.seed(10)
def visualize_data( defect_type = 'random' ,
                    image_label_tuple : tuple = None,
                    show_new_data_point = False) -> None:
    """
    defect type :str
        default_value = 'random'
        of of 10 defect types avaliable in the dataset to choose from:
         english version : punching / welding_line / crescent_gap / water_spot / oil_spot / silk_spot / inclusion
                           rolled_pit / crease / waist
         german version : 
         
    image_label_tuple: output of the __getitem__ of the dataset definition , if given the class_instance[index] 
    this function plots the image with the bounding box 
         
    show_new_data_point: bool
        default_value = True
        if value is True, the function shows a different data point each time it runs 
        if False, the functions stays on the first data point it showed 
    
    """

    if image_label_tuple:

        #[[defect_categories[defect_type],xmin,ymin,xmax,ymax], ...] 
        image_as_tensor = image_label_tuple[0].numpy()
        bounding_boxes = image_label_tuple[1]


        # Create figure and axes
        fig, ax = plt.subplots(1)
        # Display the image
        ax.imshow(image_as_tensor,cmap='gray') #cmap='gray'
        
        def get_key_by_value(dictionary, value):
            for key, val in dictionary.items():
                if val == value:
                    return key
            # Value not found
            return None

        for obj in bounding_boxes:
            name = obj[0]
            if name == 10: #10 is assigned for no defects, so no bounding boxes need to be plotted 
                continue 
            xmin = obj[1]
            ymin = obj[2]
            xmax = obj[3]
            ymax = obj[4]

            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            defect_categories = extract_unique_defect_types()
            ax.text(xmin, ymin, get_key_by_value(defect_categories,name), color='r', verticalalignment='top')
            print(f"{name:15} | ({xmin:4},{ymin:4}) | ({xmax:4},{ymax:4})")
        plt.axis('off')
        plt.title(f"sample of dataset")
        plt.show()

    else: #read from the dictionary and show a sample()

        data_frame =  construct_dataframe()

        if defect_type == 'random':
            sample = data_frame.sample()
        else:
            filtered_df = data_frame[data_frame['defect_types'].apply(lambda x: defect_type in x )]
            sample = filtered_df.sample()

        xml_path = sample['xml_path'].item()
        image_path = sample['img_path'].item()

        tree_path = xml_path
        tree = ET.parse(tree_path)
        root = tree.getroot()

        # Access and extract information from the XML elements
        filename = root.find('filename').text
        path = root.find('path').text

        objects = root.findall('object')

        image = Image.open(image_path)
        # Create figure and axes
        fig, ax = plt.subplots(1)

        # Display the image
        ax.imshow(image,cmap='gray') #cmap='gray'
        print("Name            | xmin/ymin | xmax/ymax")
        # Plot bounding boxes
        for obj in objects:
            name = obj.find('name').text
            name = assign_language(name,'eng')
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)

            # Create a Rectangle patch
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')

            # Add the patch to the axes
            ax.add_patch(rect)
            ax.text(xmin, ymin, name, color='r', verticalalignment='top')

            print(f"{name:15} | ({xmin:4},{ymin:4}) | ({xmax:4},{ymax:4})")

        # Set axis labels and title
        plt.axis('off')
        plt.title(f"sample of dataset")

        # Show the image with bounding boxes
        plt.show()



def construct_dataframe(image_directory : str = os.path.join('.','defects-class-and-location','images','images'),
                        xml_directory : str = os.path.join('.','defects-class-and-location','label','label'),
                       reduced = False) -> pd.DataFrame:
    
    """
    input parameters : 
    image_directory        : path of the dataset's images
    xml_directory (labels) : path of the xml files corresponding to the images

    
    output parameters:
    df : the dataframe consisting of the following columns (with examples) : 
    
        'img_id'         = img_01_3436789500_00004
        'img_path'       = .\defects-class-and-location\images\images\crease\img_01_3436789500_00004.jpg
        'xml_path'       = .\defects-class-and-location\label\label\img_01_3436789500_00004.xml
        'defect_types'   = ['crease', 'crease']
        'dimensions'     = [2048, 1000, 1]
        'bounding_boxes' = [[8, 981, 182, 2046, 249], [8, 478, 179, 711, 244]]
        'targets'     = [[8, 981, 182, 2046, 249], [8, 478, 179, 711, 244], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

    
    """
    
    
    list_to_construct_dataframe = []

    for folder in os.listdir(image_directory):

        for image in os.listdir(os.path.join(image_directory,folder)):

            img_path = os.path.join(image_directory, folder, image)
            xml_path = os.path.join(xml_directory, image.replace('jpg','xml'))

            if os.path.exists(img_path) and os.path.exists(xml_path):
                image_id = image.split('.')[0]
                xml_details = read_xml(xml_path)
                list_to_construct_dataframe.append([image_id,
                                                    img_path,
                                                    xml_path,
                                                    xml_details['defect_types'],
                                                    xml_details['dimensions'],
                                                    xml_details['bounding_boxes'],
                                                   ])

    df = pd.DataFrame(list_to_construct_dataframe, columns=["img_id", "img_path",'xml_path','defect_types', "dimensions",'bounding_boxes'])
    unique_defects = list(set([element for sublist in df['defect_types'] for element in sublist]))
    df['labels'] = df['bounding_boxes'].apply(lambda x: x+ [[10,0,0,0,0]]* (6 - len(x))) #the 10 is the index for no defect

    return df

def extract_unique_defect_types():
    
    #defect_types = construct_dataframe().explode('defect_types')['defect_types'].unique()
    #return {item:index for index,item in enumerate(defect_types)}
     return {'crease': 0,
             'silk_spot': 1,
             'water_spot': 2,
             'rolled_pit': 3,
             'inclusion': 4,
             'crescent_gap': 5,
             'welding_line': 6,
             'punching_hole': 7,
             'oil_spot': 8,
             'waist_folding': 9
            }


def transform_image_and_bbs(img_arr, bboxes, h, w):
    """
    :param img_arr: original image as a numpy array
    :param bboxes: bboxes as numpy array where each row is 'x_min', 'y_min', 'x_max', 'y_max', "class_id"
    :param h: resized height dimension of image
    :param w: resized weight dimension of image
    :return: dictionary containing {image:transformed, bboxes:['x_min', 'y_min', 'x_max', 'y_max', "class_id"]}
    """
    # create resize transform pipeline
    transform = albumentations.Compose(
        [albumentations.Resize(height=h, width=w, always_apply=True)],
        bbox_params=albumentations.BboxParams(format='pascal_voc'))

    transformed = transform(image=img_arr, bboxes=bboxes)

    return transformed


def create_sliding_video(image_path : str, sliding_direction = "horizontal", duration=10):
    """
    input: 
        image_path : string : image's path 
        sliding_direction : 'horizontal' or 'vertical' : decide on the frames moving axis
        duration : int : duration of the produced video 

    output : 
        .mp4 video of the sliding frame 

    example usage:
            create_sliding_video(image_path, "vertical", 10) 

    """
    # Read the image
    image = cv2.imread(image_path)
    
    # Extract image dimensions
    height, width, _ = image.shape
    
    # Set the sliding distance based on the direction
    if sliding_direction == "horizontal":
        sliding_distance = width
    elif sliding_direction == "vertical":
        sliding_distance = height
    else:
        raise ValueError("Invalid sliding direction. Please choose 'horizontal' or 'vertical'.")
    
    # Calculate the number of frames and duration per frame
    num_frames = int(duration * 30)  # Assuming 30 frames per second
    sliding_speed = sliding_distance / num_frames
    
    # Create an empty video writer object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter("sliding_video.mp4", fourcc, 30, (width, height))
    
    # Generate each frame of the sliding video
    for i in range(num_frames):
        # Calculate the offset based on the sliding direction
        if sliding_direction == "horizontal":
            offset = int(i * sliding_speed)
            frame = np.roll(image, offset, axis=1)
        elif sliding_direction == "vertical":
            offset = int(i * sliding_speed)
            frame = np.roll(image, offset, axis=0)
        
        # Write the frame to the video
        output_video.write(frame)
    
    # Release the video writer and destroy any remaining windows
    output_video.release()
    cv2.destroyAllWindows()
    
    print("Sliding video created successfully.")


def transform_image_and_bbs(img_arr, bboxes, h, w):
    """
    :param img_arr: original image as a numpy array
    :param bboxes: bboxes as numpy array where each row is 'x_min', 'y_min', 'x_max', 'y_max', "class_id"
    :param h: resized height dimension of image
    :param w: resized weight dimension of image
    :return: dictionary containing {image:transformed, bboxes:['x_min', 'y_min', 'x_max', 'y_max', "class_id"]}
    """
    # create resize transform pipeline
    transform = albumentations.Compose(
        [albumentations.Resize(height=h, width=w, always_apply=True)],
        bbox_params=albumentations.BboxParams(format='pascal_voc'))

    transformed = transform(image=img_arr, bboxes=bboxes)

    return transformed



def calculate_bbox_parameters(xmin, xmax, ymin, ymax, image_width, image_height):
    width = xmax - xmin
    height = ymax - ymin
    x_center = (xmin + xmax) / (2 * image_width)
    y_center = (ymin + ymax) / (2 * image_height)
    width /= image_width
    height /= image_height
    return x_center, y_center, width, height


def construct_yolo_compatible_data_structure(dataframe): #train or val 
    for i in ['train','val']:
        print(f'constructing dataframe for {i}')
        for index,row in dataframe.iterrows():
            #copy image to new directory
            source_file = os.path.join(os.getcwd(),row['img_path'])
            destination_file = os.path.join(os.getcwd(),'yolo_dataset',i,'images',f"{row['img_id']}.jpg")
            

            # Read the image using PIL and convert to PIL 
            image_as_numpy_arr = np.array(Image.open(source_file))
            bbs = []
            #convert bbs to shape that works with albumenations 
            for bb in row['bounding_boxes']:
                #current defect_type,xmin,ymin,xmax,ymax]) convert to  'x_min', 'y_min', 'x_max', 'y_max', "class_id"
                bbs.append([bb[1],bb[2],bb[3],bb[4],bb[0]])
            h,w=640,640
            transformed_img_and_bbs = transform_image_and_bbs(image_as_numpy_arr, np.array(bbs), h, w)
            
            transformed_img = transformed_img_and_bbs['image']            
            transformed_bbs = transformed_img_and_bbs['bboxes']            
            
            output_image = Image.fromarray(transformed_img)
            output_image.save(destination_file)

            #os.rename(current_name, new_name)
            bounding_boxes = ''

            for bb in bbs:
                category = bb[-1]
                x_center, y_center, width, height = calculate_bbox_parameters(bb[0],bb[2],bb[1],bb[3],row['dimensions'][0],row['dimensions'][1])
                bounding_boxes += f'{category} {x_center} {y_center} {width} {height}\n'

                
                
            labels_path = os.path.join(os.getcwd(),'yolo_dataset',i,'labels',f'{row["img_id"]}.txt')
            #write the bounding boxes to the new text structure 
            with open(labels_path, 'w') as file:
                file.write(bounding_boxes)
        


