# System Modules
import matplotlib.pyplot as plt
import skvideo.io
import numpy as np

# Deep Learning Modules
from torch.utils.data import Dataset

# User Defined Modules
from serde import read_config
from utils.visualization import *


class Prediction:
    '''
    This class represents prediction process similar to the Training class.

    '''
    def __init__(self,cfg_path):
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.setup_cuda()

        
    def setup_cuda(self, cuda_device_id=0):
        '''Setup the CUDA device'''
        torch.backends.cudnn.fastest = True
        torch.cuda.set_device(cuda_device_id)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
    def setup_model(self, model,model_file_name=None):
        '''
        Setup the model by defining the model, load the model from the pth file saved during training.
        
        '''
        if model_file_name == None:
            model_file_name = self.params['trained_model_name']        
        self.model_p = model().to(self.device)

        #Loads model from model_file_name and default network_output_path
#         self.model_p.load_state_dict(torch.load(self.params['network_output_path'] + '/' + model_file_name))
        self.model_p.load_state_dict(torch.load(self.params['network_output_path'] + "/epoch_" + '13' + '_' + model_file_name))
        # put the numer of the epoch inside ''

        
    def predict(self,predict_data_loader,visualize=True,save_video=False):
        # Reads params to check if any params have been changed by user
        self.params = read_config(self.cfg_path)
        #evaluation mode
        self.model_p.eval()
               
        if save_video:
            self.writer = self.create_video_writer()
            
        with torch.no_grad():
            for j, images in enumerate(predict_data_loader):
                #Batch operation: depending on batch size more than one image can be processed.
                images = images.float()                
                images = images.to(self.device)

                outputs = self.model_p(images)
                               
            #for each image in batch
                for i in range(outputs.size(0)):
                    image=images[i]/255
                    output=outputs[i]
                    output = torch.unsqueeze(output, 0)
                    overlay_image, prob_image = get_output_images(image,output)

                    #Converts image, overlay image and probability image to numpy so that it can be visualized using matplotlib functions later.
                    image = self.convert_tensor_to_numpy(image)
                    overlay_image_np = self.convert_tensor_to_numpy(overlay_image)
                    prob_image_np = self.convert_tensor_to_numpy(prob_image)

                    if save_video:
                        #Concatentates input and overlay image(along the width axis [axis=1]) to create video frame.
                        video_frame = np.concatenate((image, overlay_image_np), axis=1)
                        video_frame= video_frame * 255
                        video_frame = video_frame.astype(np.uint8)
                        #Writes video frame
                        self.writer.writeFrame(video_frame)

                    if(visualize):
                        display_output(image, prob_image_np, overlay_image_np)

            if save_video:
                self.writer.close()
                return play_notebook_video(self.params['output_data_path'], self.params['trained_model_name'])
            
            
    def create_video_writer(self):
        '''Initialize the video writer'''
        filename="outputvideo.mp4"
        #filename="outputvideo.mp4"
        output_path=self.params['output_data_path']
        self.writer = skvideo.io.FFmpegWriter(os.path.join(output_path, filename).encode('ascii'))
        return self.writer
       

    def convert_tensor_to_numpy(self,tensor_img):
        '''
        Convert the tensor image to a numpy image

        '''
        #torch has numpy function but it requires the device to be cpu
        tensor_img = tensor_img.cpu()        
        tensor_img.to(torch.device('cpu'))
        np_img = tensor_img.numpy()
        
        # np_img image is now in  C X H X W
        # transposes the array to H x W x C
        np_img = np.transpose(np_img, (1,2,0))

        return np_img
        
