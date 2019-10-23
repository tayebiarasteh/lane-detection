"""
This module contains filters that can be used to randomize the configuration
of single layers. For example, using the TiltRoadFilter, the tilt of the road
(and the lanes) of a StraightRoadLayer can be varied randomly.

@author: Sebastian Lotter <sebastian.g.lotter@fau.de>
"""
import random
import numpy as np

class ConfigFilter():
    """
    Base class for config-randomizing filters

    Every ConfigFilter has the ability to filter a given layer configuration,
    modifying parameters in a pseudo-random fashion.
    """
    def filter(self, config):
        return config


class TiltRoadFilter(ConfigFilter):
    """
    Tilts a straight road uniformly within [lb;ub].
    """
    def __init__(self, lb, ub):
        # Draw a tilt uniformly from [lb;ub] and save it as attribute
        self.tilt = np.random.uniform(lb,ub)
        
        
    def filter(self, config):
        # Use drawn tilt to tilt road and lanes by modifying the coordinates
        # defined in the 'config'
        tilt = self.tilt
        road = config['layer_params']['road']
        lanes = config['layer_params']['lanes']
        
        config['layer_params']['road'] = [[x,(1-x)*tilt+y] for [x,y] in road]
        config['layer_params']['lanes'] = [[[x,(1-x)*tilt+y] for [x,y] in lane] for lane in lanes]    
        
#         import pdb; pdb.set_trace()
        
        return config


class ShiftRoadFilter(ConfigFilter):
    """
    Shifts a straight road within [lb;ub].
    """
    def __init__(self, lb, ub):
        # Draw a shift uniformly from [lb;ub] and save it as attribute
        self.shift_road = np.random.uniform(lb,ub)
            
    def filter(self, config):
        # Use drawn shift to shift road and lanes (why lanes?) by modifying the coordinates
        # defined in the 'config'
        config['layer_params']['road'][0][0] += self.shift_road
        config['layer_params']['road'][1][0] += self.shift_road
        config['layer_params']['road'][1][1] += self.shift_road
        config['layer_params']['road'][0][1] += self.shift_road
        lanes = config['layer_params']['lanes']
        i = 0 # Counter

        for lane in lanes:
            config['layer_params']['lanes'][i][0][0] += self.shift_road
            config['layer_params']['lanes'][i][1][0] += self.shift_road
            config['layer_params']['lanes'][i][1][1] += self.shift_road
            config['layer_params']['lanes'][i][0][1] += self.shift_road

            i += 1


#         import pdb; pdb.set_trace()
    
        return config


class ShiftLanesFilter(ConfigFilter):
    """
    Shifts lanes horizontally within [x-lb;x+ub].
    """
    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub
        
    def filter(self, config):
        # For each lanes configuration in 'config'
        #   Draw a random shift from [lb;ub]
        lanes = config['layer_params']['lanes']
        i = 0 # Counter

        for lane in lanes:
            shift = np.random.uniform(self.lb, self.ub)
            config['layer_params']['lanes'][i][0][0] += shift
            config['layer_params']['lanes'][i][1][0] += shift
            i += 1
            
#             import pdb; pdb.set_trace()

        return config


class LaneWidthFilter(ConfigFilter):
    """
    Varies lane width within [width-lb;width+ub].
    """
    def __init__(self, lb, ub):
        # Draw random delta_width from [lb;ub] and save it
        self.lb = lb
        self.ub = ub

    def filter(self, config):
        # Use drawn delta_width to modify lane widths
        
        widths = config['layer_params']['lane_widths']
        shift0 = np.random.uniform(widths[0] - self.lb, widths[0] + self.ub)
        shift1 = np.random.uniform(widths[1] - self.lb, widths[1] + self.ub)
        shift2 = np.random.uniform(widths[2] - self.lb, widths[2] + self.ub)
        shift3 = np.random.uniform(widths[3] - self.lb, widths[3] + self.ub)

        config['layer_params']['lane_widths'][0] = shift0
        config['layer_params']['lane_widths'][1] = shift1
        config['layer_params']['lane_widths'][2] = shift2
        config['layer_params']['lane_widths'][3] = shift3
        
#         import pdb; pdb.set_trace()

        return config


class ConstantColorFilter(ConfigFilter):
    """
    Picks random color from ([r-dr;r+dr],[g-dg;g+dg],[b-db;b+db]) to vary color
    of constant color function.
    """
    def __init__(self, dr, dg, db):
        # Draw delta_r/g/b and save them
        self.dist_r = random.randint(-1*dr, dr)
        self.dist_g = random.randint(-1*dg, dg)
        self.dist_b = random.randint(-1*db, db)

    def filter(self, config):
        # Modify color defined in the config
        config['layer_params']['color_fct']['params']['color'][0] += self.dist_r
        config['layer_params']['color_fct']['params']['color'][1] += self.dist_g
        config['layer_params']['color_fct']['params']['color'][2] += self.dist_b

        return config


class RandomColorMeanFilter(ConfigFilter):
    """
    Picks random color from ([r-dr;r+dr],[g-dg;g+dg],[b-db;b+db]) to vary mean
    of random color function.
    """
    def __init__(self, dr, dg, db):
        # Draw delta_r/g/b and save them
        self.dist_r = random.randint(-1*dr, dr)
        self.dist_g = random.randint(-1*dg, dg)
        self.dist_b = random.randint(-1*db, db)

    def filter(self, config):
        # Modify color defined in the config
        config['layer_params']['color_fct']['params']['mean'][0] += self.dist_r
        config['layer_params']['color_fct']['params']['mean'][1] += self.dist_g
        config['layer_params']['color_fct']['params']['mean'][2] += self.dist_b

        return config


# Public API
# Exporting a registry instead of the functions allows us to change the
# implementation whenever we want.
CONFIG_FILTER_REGISTRY = {
    'ShiftRoadFilter'       : ShiftRoadFilter,
    'ShiftLanesFilter'      : ShiftLanesFilter,
    'TiltRoadFilter'        : TiltRoadFilter,
    'LaneWidthFilter'       : LaneWidthFilter,
    'ConstantColorFilter'   : ConstantColorFilter,
    'RandomColorMeanFilter' : RandomColorMeanFilter
}


