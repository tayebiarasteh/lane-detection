#System Modules
import os.path
from enum import Enum
import datetime

# Deep Learning Modules
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
import torch

# User Defined Modules
from serde import *
from utils.visualization import get_output_images
# from utils.augmentation import *

class Training:
    '''
    This class represents training process.
    Various functionalities in the training process such as setting up of devices, defining model and its parameters,
    executing training can be found here.
    '''
    def __init__(self,cfg_path,torch_seed=None):
        '''
        Args:
            cfg_path (string): 
                path of the experiment config file

            torch_seed (int):
                Seed used for random gererators in pytorch functions
                
            dataset_parent_path (string):
        '''
        
        self.params=read_config(cfg_path)
        self.cfg_path=cfg_path
        self.model_info=self.params['Network']
        self.model_info['seed']=torch_seed or self.model_info['seed']
        if  'trained_time' in self.model_info:
            self.raise_training_complete_exception()

        self.setup_cuda()
        self.writer = SummaryWriter(os.path.join(self.params['tf_logs_path']))


    def setup_cuda(self, cuda_device_id=0):
        '''Setup the CUDA device'''
        torch.backends.cudnn.fastest = True
        torch.cuda.set_device(cuda_device_id)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.manual_seed_all(self.model_info['seed'])
        torch.manual_seed(self.model_info['seed'])

    def setup_model(self, model, optimiser,optimiser_params,loss_function,weight_ratio):
        '''
        Setup the model by defining the model, optimiser,loss function and learning rate.
        '''
        #Tensor Board Graph
        self.add_tensorboard_graph(model)
        self.model = model().to(self.device)

        # optimiser_params is a dictionary containing parameters for the optimiser.
        self.optimiser = optimiser(self.model.parameters(), **optimiser_params)

        self.loss_weight=torch.tensor([1,weight_ratio],dtype=torch.float).to(self.device) #Non lane is class 0 and lane is class 1
        self.loss_function = loss_function(weight=self.loss_weight)

        if 'retrain' in self.model_info and self.model_info['retrain']==True:
            self.load_pretrained_model()
            
        # CODE FOR CONFIG FILE TO RECORD MODEL PARAMETERS
        #Saves the model, optimiser,loss function name for writing to config file
        self.model_info['model_name']=model.__name__
        self.model_info['optimiser']=optimiser.__name__
        self.model_info['loss_function']=loss_function.__name__
        self.model_info['optimiser_params']=optimiser_params
        self.model_info['lane_to_nolane_weight_ratio']=weight_ratio
        self.params['Network']=self.model_info
        write_config(self.params, self.cfg_path,sort_keys=True)
        
       
    def add_tensorboard_graph(self,model):
        '''
        Creates a tensor board graph for network visualisation
        '''
        # Tensorboard Graph for network visualization
        
        dummy_input = torch.rand(1, 3, 256, 256)  # To show tensor sizes in graph
        self.writer.add_graph(model(), dummy_input)

    def execute_training(self,train_loader,test_loader,num_epochs=None):
        '''
        Execute training by running training and validation at each epoch
        '''
        self.steps=0

        # reads param file again to include changes if any
        self.params=read_config(self.cfg_path)
        
        #Checks if already trained
        if  'trained_time' in self.model_info:
            self.raise_training_complete_exception

        # CODE FOR CONFIG FILE TO RECORD MODEL PARAMETERS
        self.model_info = self.params['Network']
        self.model_info['num_epochs']=num_epochs or self.model_info['num_epochs']
              
        self.epoch = 0
        print('Starting time:' + str(datetime.datetime.now()))
        for epoch in range(num_epochs):
            self.epoch = epoch
            print('')
            print('Training:')
            self.train_epoch(train_loader)
            print('')
            print('Testing:')
            self.test_epoch(test_loader)
      
        # Saves model using path from self.params['network_output_path'] and self.params['trained_model_name']
            torch.save(self.model.state_dict(), self.params['network_output_path'] + "/epoch_" + str(self.epoch) + '_' + self.params['trained_model_name'])
#             torch.save(self.model.state_dict(), self.params['network_output_path'] + "/" + self.params['trained_model_name'])

        #CODE FOR CONFIG FILE TO RECORD TRAINING INFO
        #Saves information about training to config file
        self.model_info['num_steps']=self.steps
        self.model_info['trained_time']="{:%B %d, %Y, %H:%M:%S}".format(datetime.datetime.now())
        self.params['Network'] = self.model_info
        
        write_config(self.params, self.cfg_path,sort_keys=True)

    def train_epoch(self,train_loader):
        '''Train using one single iteratation of all images(epoch) in dataset'''
        print("Epoch [{}/{}] \n".format(self.epoch +1, self.model_info['num_epochs']))
        #Create list to store loss value to display statistics
        loss_list = []
      
        for i, (image, label) in enumerate(train_loader):
            image = image.float()
            label = label.long()
            image = image.to(self.device)
            label = label.to(self.device)
            
            #Forward pass.
            self.optimiser.zero_grad()
            
            with torch.set_grad_enabled(True):                
                output = self.model(image)
                loss = self.loss_function(output, label)
                loss_list.append(loss.data[0])

                #Backward and optimize
                loss.backward()
                self.optimiser.step()

                # Prints loss statistics after certain number of steps as specified in params['display_stats_freq'].
                if (i+1)%self.params['display_stats_freq'] == 0:
                    self.calculate_loss_stats(loss_list)

                    #reset loss list after number of steps as specified in params['display_stats_freq']
                    loss_list= []
            self.steps = self.steps + 1


    def test_epoch(self,test_loader):
        '''Test model after an epoch and calculate loss on test dataset'''
        self.model.eval()

        #Initializes an image list for tensor board
        img_list = []
        with torch.no_grad():
            loss_list = []
            for i, (image, label) in enumerate(test_loader):
                image = image.float()
                label = label.long()

                image = image.to(self.device)
                label = label.to(self.device)

                output = self.model(image)
                loss = self.loss_function(output, label)
                loss_list.append(loss.data[0])

                if i % 5 == 0:
                    # overlay image and probability image from function in visualization.py
                    # Image tensors must be between 0 to 1 and not 0 to 255 for this function. Also, you require input in  CxHxW dimensions not NxCxHxW

                    image = torch.squeeze(image, 0)
                    image=image/255
                    overlay_img,prob_img= get_output_images(image,output)
                    overlay_img = (overlay_img*255).float().to(self.device)
                    prob_img = (prob_img*255).float()
                    prob_img =prob_img.to(self.device)
                    images=image*255
                    img_list.extend([images, prob_img, overlay_img])
                    self.calculate_loss_stats(loss_list,is_train=False)

        #Setup Tensorboard Image
        image_grid = make_grid(img_list, nrow=3)
        self.writer.add_image('Input-Segmentation Probabilty-Overlay', image_grid, self.steps)
        self.model.train()


    def calculate_loss_stats(self,loss_list,is_train=True):
        # Converts list to tensor inorder to use other torch functions to calculate statistics later
        loss_list = torch.stack(loss_list)
        avg = torch.mean(loss_list)
              
        #Print result to tensor board and std. output 
        if is_train:
            mode='Train'
        else:
            mode='Test'
            
        #Adds average loss value to tensorboard 
        self.writer.add_scalar(mode+'_Loss', avg, self.steps)

        # Prints stats
        print('average value of the losses: ', avg)
        
    def load_pretrained_model(self):
        '''Load pre trained model to the using  pretrain_model_path parameter from config file'''
        self.model.load_state_dict(torch.load(self.model_info['pretrain_model_path']))
        
    def raise_training_complete_exception(self):
        raise Exception("Model has already been trained on {}. \n"
                                            "1.To use this model as pre trained model and train again\n "
                                            "create new experiment using create_retrain_experiment function.\n\n"
                                            "2.To start fresh with same experiment name, delete the experiment  \n"
                                            "using delete_experiment function and create experiment "
                            "               again.".format(self.model_info['trained_time']))


class Mode(Enum):
    '''
    Class Enumerating the 3 modes of operation of the network.
    This is used while loading datasets
    '''

    TRAIN = 0
    TEST = 1
    PREDICT = 2


