import os
import pickle
import fnmatch
from skimage.io import imread
from skimage.transform import resize
from skimage.util import img_as_ubyte
import numpy as np
import datetime
from torch.utils.data import Dataset
import torch
from specs import *
from serde import read_detection, read_config,write_config
from pipelines import simulation_pipeline
from Training import Mode
from pipelines import label_true_negatives
from config import sim_config
import random
import re

sc = sim_config()['simulator']

HEIGHT=sc['height']
WIDTH=sc['width']
DEFAULT_DATA_PATH=read_config('./config.json')['input_data_path']

class Img_dataset(Dataset):
    """
    Class representing Image Dataset
    This class is used to represent both Simulation datasets and true negative real world image datasets.
    The images are also returned in the HEIGHT and WIDTH obtained from the simulator config.
    Depending on the mode specified by the user, the Dataset returns labels for train and test modes.
    User also has the option of choosing smaller sample from a folder containg large number of images by setting the size 
    parameter    
    """
    DEFAULT_DATA_PATH = read_config('./config.json')['input_data_path']

    def __init__(self, dataset_name, size,cfg_path, mode=Mode.TRAIN, dataset_parent_path=DEFAULT_DATA_PATH
                 , augmentation=None, seed=1):
        """
        Args:
            dataset_name (string): Folder name of the dataset.
            size (int):
                No of images to be used from the dataset folder
            mode (enumeration Mode):
                Nature of operation to be done with the data.
                Possibe inputs are Mode.PREDICt,Mode.TRAIN,Mode.TEST
                Default value: Mode.TRAIN
            dataset_parent_path (string):
                Path of the folder where the dataset folder is present
                Default: DEFAULT_DATA_PATH as per config.json
            cfg_path (string):
                Config file path of your experiment

            augmentation(Augmentation object):
                Augmentation to be applied on the dataset. Augmentation is passed using the object
                from Compose class (see augmentation.py)
            seed
                Seed used for random functions
                Default:1

        """
        params = read_config(cfg_path)
        self.cfg_path=params['cfg_path']
        self.detections_file_name = params['detections_file_name']
        self.mode = mode
        self.dataset_path=os.path.join(dataset_parent_path,dataset_name)
        self.datset_name=dataset_name
        self.size=size
        self.augmentation = augmentation
        self.dataset_parent_path = dataset_parent_path
        self.img_list = self._init_dataset(dataset_name,seed,params)
       

    def __len__(self):
        '''Returns length of the dataset'''
        return self.size


    def __getitem__(self, idx):
        '''
        Using self.img_list and the argument value idx, return images and labels(if applicable based on Mode)
        The images and labels are returned in torch tensor format.
        '''

        #Reads images using files name availble in self.img_list
        img = imread(self.img_list[idx])
        img = resize(img, (HEIGHT, WIDTH))
        
        #Conversion to ubyte value range (0...255) is done here, because network needs to be trained and needs to predict using the same datatype.
        img = img_as_ubyte(img)

        if self.mode == Mode.PREDICT:
            img = img.transpose((2, 0, 1))                   
            img = torch.from_numpy(img)            
            return img
        
        #If mode is not PREDICT, Obtains binary label image 
        else:
            label_point_cloud = read_detection(os.path.join(self.dataset_path , self.detections_file_name), os.path.basename(self.img_list[idx]))
            x = label_point_cloud[0] ; y = label_point_cloud[1] 
            label = np.zeros((img.shape[0],img.shape[1]))
            label[x, y]=1
                          
        #Apply augmentation if applicable

            #Converts image and label to tensor and returns image,label
            img = img.transpose((2, 0, 1))
            img = torch.from_numpy(img)
            label = torch.from_numpy(label)
            return img, label


    def _init_dataset(self,dataset_name,seed,params):
        '''
        If the dataset is found , the size is checked against the number of images in the folder and
        if size is more than number of images in the folder, randomly pick images from the folder so
        that number of images is equal to the user specified size.

        If the dataset is not found, the simulator is run and a folder containing simulation
        images and detection files, is created with the given dataset name.

        Final image list is stored into self.img_list
        '''

        # Checks if the dataset directory exists
        if os.path.isdir(self.dataset_path):
            # If dataset directory found: Collects the file names of all images and stores them to a list
            image_list = os.listdir(self.dataset_path)
            img_list = []
            for item in image_list:
                if ((re.search(".png", str(item))) or (re.search(".jpeg", str(item))) or (re.search(".jpg", str(item)))):
                    img_list.append(os.path.join(self.dataset_path, item))
            
            if len(img_list) < self.size:
                print('The size you requested is more than the total available images.')
                self.size = len(img_list)
                self.img_list = img_list

            elif len(img_list) > self.size:
                print('The size you requested is less than the total available images. The desired number of images randomly will be picked.')
                random.seed(seed)
                random.shuffle(img_list)
                self.img_list = []
                self.img_list = img_list[:self.size]
            # would contain number of images as specified by user.

            elif len(img_list) == self.size:
                self.img_list = img_list

        # If dataset directory not found: Runs the simulator and obtain img list from simulation dataset.
        else:
            self.dataset_path = simulation_pipeline(params, self.size, dataset_name, seed)
            image_list = os.listdir(self.dataset_path)
            self.img_list = []
            for item in image_list:
                if ((re.search(".png", str(item))) or (re.search(".jpeg", str(item))) or (re.search(".jpg", str(item)))):
                    self.img_list.append(os.path.join(self.dataset_path, item))

        # Checks if detection file is present in the folder and if it is not present creates a true 
        # negative detection file using label_true_negatives function from pipelines.py 
        #will be done from pipeline, through the point cloud.
        if not os.path.isfile(os.path.join(self.dataset_path , self.detections_file_name)):
            label_true_negatives(self.dataset_path, self.detections_file_name)
       
        # CODE FOR CONFIG FILE TO RECORD DATASETS USED
        # Saves the dataset information for writing to config file
        if self.mode==Mode.TRAIN:
            params = read_config(self.cfg_path)
            params['Network']['total_dataset_number']+=1
            dataset_key='Traing_Dataset_'+str(params['Network']['total_dataset_number'])
            #If augmentation is applied
            if self.augmentation:
                augmenetation_applied=[i.__class__.__name__ for i in self.augmentation.augmentation_list]
            else:
                augmenetation_applied=None
            dataset_info={
                'name':dataset_name,
                'path':self.dataset_path,
                'size':self.size,
                'augmenetation_applied':augmenetation_applied,
                'seed':seed
            }
            params[dataset_key]=dataset_info
            write_config(params, params['cfg_path'],sort_keys=True)          
        return self.img_list

