"""
Some utility functions for the image simulator.

@author: Sebastian Lotter <sebastian.g.lotter@fau.de>
"""
import numpy as np
from skimage import transform as tf
from skimage.draw import polygon


def project(coords, tform):
    fmask_proj_idx = tform(coords)
    return fmask_proj_idx


def init_transform(src, tgt):
    tform = tf.ProjectiveTransform()
    tform.estimate(
        np.array(src),
        np.array(tgt)
    )

    return tform


def scale_coords(coordinates, height, width):
    return coordinates * np.array([height, width])


def draw_polygon(img_height, img_width, vertices):
    """
    Uses scikit-image draw polygon function to draw a polygonal
    shape on an empty image

    Input:
    img_height -- image height
    img_width  -- image width
    vertices   -- List of vertices in relative coordinates, e.g. 
        [(0.0,0.3), (0.0,0.7), (0.3,0.5), (0.0,0.3)] will create a reversed triangle
        on the top of the image
        Attention: Vertices must be in order as they shall be drawn, e.g. top-left, top-right, bottom-right, bottom-left

    Output:
    bitmask representing the drawn shape
    """
    fmask = np.zeros((img_height, img_width), dtype=np.uint8)
    r = np.round(img_height * np.array(vertices)[:,0])
    c = np.round(img_width * np.array(vertices)[:,1])
    rr, cc = polygon(r, c, (img_height,img_width))
    fmask[rr, cc] = 1 

    return fmask


def merge_layers(layers):
    img = np.zeros((layers[0].fmask.shape) + (3,), dtype=np.uint8)

    for layer in layers:
        rendered_layer = layer.render()
        idx = layer.fmask > 0
        img[idx,:] = rendered_layer[idx,:]

    return img

