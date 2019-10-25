"""
Simulation pipeline and some helper functions.

@author: Sebastian Lotter <sebastian.g.lotter@fau.de>
"""
import numpy as np
import os
import os.path
import random

# User imports
from config import sim_config
from serde import write_detections, write_img, write_config
from simulator.simulator import simulate_road_img


DEFAULT_SEED = 42
DEFAULT_CONFIG_FILENAME = 'config.json'


def init_rng(seed):
    """
    To make results reproducible fix seed of RNG
    """
    random.seed(seed)


def simulation_pipeline(params, batch_size, dataset_name, seed):
    """
    Main loop of the image simulator. Simulates images and saves them
    to the input data folder together with the configuration and
    the corresponding detections.

    Input:
    params       -- global parameters like paths, filenames etc.
    batch_size   -- number of images to simulate
    dataset_name -- name of the dataset to be produced
    seed         -- seed for the RNG (optional)
    """
    init_rng(seed or DEFAULT_SEED)

    dataset_path = setup_data_dir(params, dataset_name)
    sim_cfg = sim_config()
    detections = {}

    # Serialize config to dataset folder
    write_config({
        'global': params,
        'simulator': sim_cfg,
        'others': {
            'seed': seed
        }
    }, os.path.join(dataset_path, DEFAULT_CONFIG_FILENAME))

    # Simulator main loop
    for i in range(batch_size):
        # Simulate image
        img, detection = simulate_road_img(sim_cfg['simulator'])

        # Store results
        filename = params['img_file_prefix'] + str(i) + params['img_file_suffix']
        write_img(img, os.path.join(dataset_path, filename))
        add_detection(detections, filename, detection=detection)

    write_detections(detections, os.path.join(dataset_path, params['detections_file_name']))
    
#     self.datasetpath = dataset_path
    
    return dataset_path


def setup_data_dir(params, dataset_name):
    dataset_path = os.path.join(os.path.abspath(params['input_data_path']), dataset_name)

    try:
        os.makedirs(dataset_path)
    except OSError as e:
        raise Exception(("Directory %s already exists. Please choose an unused dataset " +
                "identifier") % (dataset_path,))

    return dataset_path


def add_detection(detections, id, detection):
    detections[str(id)] = detection


def label_true_negatives(dataset_path, detections_file_name):
    """
    In order to include real world images, that do not contain roads, in the training,
    they are treated as regular dataset without detections/on-road pixels.
    """
    detections = {}
    for f in os.listdir(dataset_path):
        add_detection(detections, f, detection=np.array([[],[],[],[],[]], dtype=np.int32))

    write_detections(detections, os.path.join(dataset_path, detections_file_name))

