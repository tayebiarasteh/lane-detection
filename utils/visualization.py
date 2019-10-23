import matplotlib.pyplot as plt
from IPython.display import Markdown as md
from torchvision import transforms
import torch
from PIL import Image
import PIL
import torchvision
import os
import numpy as np
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def get_probability_map(output_3d):
    '''Returns the probabilty map tensor(2d: H * W) for the given network-output tensor (3d:C* H * W)'''
    
    #use function softmax2d
    
    softmaxfunc = nn.Softmax2d()
    return softmaxfunc(output_3d)
    


def get_output_images(input_layer,output_3d):
    ''' Returns overlayed image and probabilty map in 3d: C * H * W
        Inputs:
            input_layer: 3D input image. Value range between 0 to 1.
            output_3d: 3D tensor of network outputs '''
    # Generate the probabilty map using above function get_probability_map(). Use network output tensor as input.
    #Use round/threshold function to assign pixels as 0 (no detection) or 1 (detection) 
    # create a 3 channel image with probabilty map in the red channel.

    prob_Image = get_probability_map(output_3d)
    prob_Image= torch.squeeze(prob_Image, 0)
    prob_image = prob_Image[1]
    prob_image=prob_image.cpu()
    prob_image_R= torch.unsqueeze(prob_image, 0)
    prob_image_R= prob_image_R.cpu()
    prob_image_R = torch.cat((prob_image_R, torch.zeros(2, 256, 256)))
    prob_image_D = prob_image_R.numpy()
    prob_image_D=np.where(prob_image_D < 0.5, prob_image_D, 1)
    prob_image_D=np.where(prob_image_D >= 0.5, prob_image_D, 0)
    prob_image_N = prob_image.numpy()
    prob_image_N=np.where(prob_image_N < 0.5, prob_image_N, 1)
    prob_image_N=np.where(prob_image_N >= 0.5, prob_image_N, 0)
    input_layer= input_layer.cpu()
    overlay_image = input_layer.numpy()
    idx = np.where(prob_image_N==1)
    overlay_image[0,:,:][idx]= 1
    overlay_image[1,:,:]= overlay_image[1,:,:] *(np.ones(prob_image_N.shape)-prob_image_N)
    overlay_image[2,:,:]= overlay_image[2,:,:] *(np.ones(prob_image_N.shape)-prob_image_N)
#     overlay_image = np.transpose(overlay_image, (2,0,1))
#     prob_image_D = np.transpose(prob_image_D, (2,0,1))
    overlay_image = torch.from_numpy(overlay_image)
    prob_image_D = torch.from_numpy(prob_image_D)
    
    
    #Overlay the class image on top of the input image to have red colour
    #at pixels where class value =1
    # You may use other modules to create the images but 
    #the return value must be 3d tensors

    return overlay_image,prob_image_D



def play_notebook_video(folder,filename):
    '''Plays video in ipython using markdown'''
    file_path=os.path.join(folder, filename)  
    return md('<video controls src="{0}"/>'.format(file_path))




def display_output(image,prob_img,overlay_img):
    '''
    Displays the output using matplotlib subplots
    
    '''
    #Inputs are numpy array images.
    plt.subplot(131)
    plt.imshow(image)
    plt.title('Original image')
    plt.subplot(132)
    plt.imshow(prob_img)
    plt.title('Probabilty map')
    plt.subplot(133)
    plt.imshow(overlay_img)
    plt.title('Overlaid image')
    plt.show()



    
 



    
    
