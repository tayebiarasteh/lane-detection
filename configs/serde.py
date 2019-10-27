"""
functions for writing/reading data to/from disk
"""
import h5py
import json
import numpy as np
from skimage.io import imsave
import os
import warnings
import shutil



"""
SPEC
def read_video():
    # read video as a list of images
    return iterator_over_img_list

def read_imgs():
    # read a folder with images (possibly only one image there)
    return iterator_over_img_list

def write_video(img_list):
    # write a list of images as a video
    pass

def write_imgs(img_list):
    # write a list of images to a folder
    pass
"""



def write_img(img, filename):
    """
    Write image to disk.
    """
    imsave(filename, img)


def write_detections(detections, filename):
    """
    Write file that contains detections in efficient hdf format.
    Layout:
    { 'some_img.png': np.array((5,N)),
      'anot_img.png': np.array((5,N)) }
    """
    with h5py.File(filename, 'w') as f:
        for d in detections:
            # Compression factor with 'poor, but fast' lzf compression almost 
            # factor 10 for 1 dataset, ~factor 35 for 100 datasets
            f.create_dataset(d, data=detections[d], compression='lzf')
            

def read_detection(filename, dataset_key):
    """
    Read detections from file.
    """
    with h5py.File(filename, 'r') as f:
        ds = f[dataset_key]
        detection = np.zeros(ds.shape, ds.dtype)
        if ds.size != 0:
            ds.read_direct(detection)

    return detection


def read_config(cfg_path):
    with open(cfg_path,'r') as f:
        params = json.load(f)

    return params


def write_config(params, cfg_path,sort_keys=False):
    with open(cfg_path,'w') as f:
        json.dump(params, f, indent=2,sort_keys=sort_keys)
        
def create_experiment(experiment_name):
    params=read_config('./config.json')
    params['experiment_name']=experiment_name
    create_experiment_folders(params)
    cfg_file_name=params['experiment_name']+'_config.json'
    cfg_path=os.path.join(params['network_output_path'],cfg_file_name)
    params['cfg_path']=cfg_path
    write_config(params, cfg_path)
    
    return params

def create_experiment_folders(params):
    try:
        path_keynames=["network_output_path","output_data_path","tf_logs_path"]
        for key in path_keynames:
            params[key]=os.path.join(params[key],params['experiment_name'])
            os.makedirs(params[key])
    except:
        raise Exception("Experiment already exist. Please try a different experiment name")
        
def open_experiment(experiment_name):
    '''Open Existing Experiments'''
    
    default_params=read_config('./config.json')
    cfg_file_name=experiment_name+'_config.json'
    cfg_path=os.path.join(default_params['network_output_path'],experiment_name,cfg_file_name)
    params=read_config(cfg_path)

    return params
    
def delete_experiment(experiment_name):
    '''Delete Existing Experiment folder'''
    
    default_params=read_config('./config.json')
    cfg_file_name=experiment_name+'_config.json'
    cfg_path=os.path.join(default_params['network_output_path'],experiment_name,cfg_file_name)

    params=read_config(cfg_path)
        
    path_keynames=["network_output_path","output_data_path","tf_logs_path"]
    for key in path_keynames:
            shutil.rmtree(params[key])


def create_retrain_experiment(experiment_name,source_pth_file_path):
    params=create_experiment(experiment_name)
    params['Network']['retrain']=True

    destination_pth_file_path=os.path.join(params['network_output_path'],'pretrained_model.pth')
    params['Network']['pretrain_model_path'] = destination_pth_file_path
    shutil.copy(source_pth_file_path,destination_pth_file_path)

    write_config(params, params['cfg_path'])
    return params
