"""
The simulator is intended to create artificial images of a cartoon-like road
together with the information which pixels belong to a lane.

This data may then be used to perform supervised learning with a neural network
for lane recognition.
"""
from copy import deepcopy
import numpy as np
import random

import sys
sys.path.append('../')
# User imports
from colors import COLOR_FCT_REGISTRY
from filters import CONFIG_FILTER_REGISTRY
from layers import LAYER_REGISTRY
from utils import merge_layers


def simulate_road_img(config):
    """
    Main function of the simulator. Creates an artificial cartoon-like image and a corresponding
    point cloud of on-lane pixels.

    An image consists of several overlayed layers, each one of them having a particular shape
    and beeing able to render itself with a configured color function (see colors.py). The
    configuration of the various layers (e.g. background layer, road layer) and their composition
    is completely up to the user. A example configuration is provided in config.py.

    In order to introduce sufficient variance for the neural net to learn in a meaningful manner,
    there exist 'filters' (see filters.py) that vary the layer configuration in a pseudo-random
    fashion. They tilt, for example, the road or shift it randomly. An example configuration of
    such filters can also be found in config.py.

    Input:
    config     -- configuration of the image layers, see config.py for an example configuration

    Output:
    img        -- the artificial image
    detections -- a listing of on-lane pixels
    """
    layers = []
    detections = np.ndarray((5,0), dtype=np.int32)
    params = deepcopy(config)

    # For each layer configured in the config
    for layer in params['layers']:
        # If it has no 'prob' attribute, or the probablity of appearing in the image
        # is bigger than a drawn random number
        if 'prob' not in layer or layer['prob'] > random.random():
            try:
                # Get the configuration for this layer
                layer_config = params['layer_configs'][layer['config_id']]
            except KeyError:
                raise 

            for f in layer.get('filters', []):
                # Apply the filters defined for this layer
                layer_config = CONFIG_FILTER_REGISTRY[f['type']](**f['params']).filter(layer_config)

            try:
                # Initialize layer
                img_layer = LAYER_REGISTRY[layer_config['layer_type']](params['height'], params['width'], **layer_config['layer_params'])

                # Add layer to layer stack
                layers.append(img_layer)

                # If the layer should be serialized, add its features to the list of detections
                if layer.get('serialize'):
                    detections = img_layer.to_point_cloud()
            except KeyError:
                raise 

    # Merge image
    img = merge_layers(layers)

    return img, detections


if __name__ == '__main__':
    pass
