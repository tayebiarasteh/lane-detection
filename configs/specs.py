# coding=utf-8
import os
import numpy as np
import random


# Data structures:

# Images:
# We will work with images of size 300*600
"""
img_w = 600
img_h = 300
img = np.array(img_h, img_w, 3)
"""

# x indexes the width of the image
# y indexes the height of the image
# img[y, x, :] is the pixel in position y, x

# Detections:
# We will store detections as point clouds:

"""
detections = np.array(N, 5)
"""

# detections[:,0] … y’s
# detections[:,1] … x’s
# detections[:,2] … t’s
# Above, t is the frame number in the video, when we are working with pictures t = -1
# detections[:,3] … network’s confidence [0:1]
# detections[:,4] … lane id: lane number (index from left to right in the image, 0,1,2 etc)


# Folder structure:
# data/input_data/img_(dataset_name)/[img1.png, img2.png]
# data/input_data/vid_(dataset_name)/vid.mpg
# data/output_data/experiment_name/img_(dataset_name)/[imgs/[img1.png, img2.png] , detections.pickle]
# data/output_data/experiment_name/vid_(dataset_name)/[output.mpg OR/AND imgs/[img1.png, img2.png] OR/AND detections.pickle]
# data/network_data/network_experiment_name/

# Basic functions:

# data.py:
def detections_to_mask(detections):
# assume that the input are detections corresponding to one frame
    return mask

# viz.py:
# functions for visualisation
def viz_mask(img, mask):
# mask is the binary mask np.array(img_h, img_w)
#put mask onto image with semi-transparent red
    return img

# network.py
def create_network():
    # create neural network in pytorch
    return network

def train(params, network):
    return network


def detect_images(iterator_over_img_list):
    return detections


# Later:

# - What is the intercept theorem mentioned by Sebastian? I know homography.
# - Splines are a very good idea!
# - Cost quality control
# - discuss PNG vs SVG vs python representation
# - combining the results of different groups

# Goals:
# - 2d simulator
# - 2d network trainded on set of artificial images
# - The network works well on new artificial images with the same statistics
# - The network works reasonably well on simple real images

# - 3d simulator
# - 3d network trainded on set of artificial videos
# - 3d network is tested on real video

def simulate_road_vid():
# creates a video, which is a collection of images
    return img_list, detections
