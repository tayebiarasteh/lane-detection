"""
Coloring functions used to create a 3-channel matrix representing an RGB image
from a layer represented by a bitmask. The output matrix should support 8 bit
color depth (i.e. its data type is uint8).
@first author: Sebastian Lotter <sebastian.g.lotter@fau.de>
"""
import numpy as np
import random


def color_w_constant_color(fmask, color):
    """
    Color whole layer with a constant color.

    Input:
    fmask -- binary mask indicating the shape that is to be coloured
    color -- (r,g,b) tuple

    Output:
    matrix of dimensions (x,y,3) representing 3-channel image
    """

    img = np.zeros((fmask.shape[0], fmask.shape[1], 3), dtype=np.int32)    
    idx = np.where(fmask!=0)    
    for i in range(len(idx[0])):
        img[idx[0][i], idx[1][i],0]=color[0]
        img[idx[0][i], idx[1][i],1]=color[1]
        img[idx[0][i], idx[1][i],2]=color[2]
    return img


def color_w_random_color(fmask, mean, range):
    """
    Colors layer with constant color, then draws random integer uniformly from
    [mean-range;mean+range] and adds it to the image.

    Input:
    fmask -- binary mask indicating the shape that is to be coloured. Is 1 at positions which shall be coloured and 0 elsewhere.
    mean  -- mean color, (r,g,b) tuple
    colorRange -- range within which to vary mean color

    Output:
    matrix of dimensions (x,y,3) representing 3-channel image
    """

    img_copy = np.zeros((fmask.shape[0], fmask.shape[1], 3), dtype = np.int32)
    idx = np.where(fmask!=0)
    for i in np.arange(len(idx[0])):
        img_copy[idx[0][i], idx[1][i],0]=mean[0]
        img_copy[idx[0][i], idx[1][i],1]=mean[1]
        img_copy[idx[0][i], idx[1][i],2]=mean[2]
    
    # random integer noise uniformly drawn from [-range;range] covering the whole image.
    unif = np.random.randint(-range, range, img_copy.shape)
    img_copy = img_copy + unif   
    return np.array(img_copy, dtype=np.uint8)


def color_w_constant_color_random_mean(fmask, mean, lb, ub):
    """
    Picks a random color from ([mean[0]-lb;mean[0]+ub],...) and colors
    layer with this color.

    Input:
    fmask -- binary mask indicating the shape that is to be coloured
    mean  -- mean color, (r,g,b) tuple
    lb    -- lower bound for the interval to draw the random mean color from
    ub    -- upper bound for the interval to draw the random mean color from

    Output:
    matrix of dimensions (x,y,3) representing 3-channel image
    """

    color = [0,0,0]
    color[0] = np.random.randint(low=mean[0]+lb, high=mean[0]+ub)
    color[1] = np.random.randint(low=mean[1]+lb, high=mean[1]+ub)
    color[2] = np.random.randint(low=mean[2]+lb, high=mean[2]+ub)
    img = np.zeros((fmask.shape[0], fmask.shape[1], 3), dtype=np.int32)   
    idx = np.where(fmask!=0)
    for i in range(len(idx[0])):
        img[idx[0][i], idx[1][i],0]=color[0]
        img[idx[0][i], idx[1][i],1]=color[1]
        img[idx[0][i], idx[1][i],2]=color[2]
    return img


# Public API
# Exporting a registry instead of the functions allows us to change the
# implementation whenever we want.
COLOR_FCT_REGISTRY = {
    'constant'            : color_w_constant_color,
    'random'              : color_w_random_color,
    'constant_random_mean': color_w_constant_color_random_mean
}

