import os

import fnmatch
from skimage.io import imread
from skimage.transform import resize
from skimage.util import img_as_ubyte
import skvideo.io
import numpy as np



from torch.utils.data import Dataset
import torch
from skimage.transform import downscale_local_mean

from config import sim_config
from pipelines import simulation_pipeline

params = {
'input_data_path' : './data/input_data'
}
sc = sim_config()['simulator']

MAX_SIZE=256
class Video_dataset(Dataset):
    """Video Dataset."""

    def __init__(self,dir=params['input_data_path'],folder_name='Video',video_file_name=None ):


        self.length=0
        folder_path = os.path.join(dir,folder_name)
        search_pattern=video_file_name or '*.mp4'
        
        for file_name in os.listdir(folder_path):
            if fnmatch.fnmatch(file_name,search_pattern):
                self.file_name=os.path.join(folder_path,file_name)
                self.videogen = skvideo.io.vreader(self.file_name)
                self.vid=list(self.videogen)
                self.length=len(self.vid)
        if self.length==0:
            raise IOError('{0} Video not found in {1}'.format(video_file_name or '',folder_path))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img=self.vid[idx]

        resized = resize(img,(sc['height'],sc['width'],3))
        img=img_as_ubyte(resized)

        # numpy image: H x W x C
        # torch image: C X H X W
        img = img.transpose((2,0,1))
        img = torch.from_numpy(img).float()
        return img






