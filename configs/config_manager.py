# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 21:33:29 2018

@author: msabih
"""

from config import sim_config
from copy import deepcopy
class ConfigManager():
    def __init__(
        self,
        params,
    ):
        self.params = params
        self.layer_set_all = set(['BackgroundLayer', 'SkyLayer',
                              'StraightRoadLayer'])
    
        self.filter_set_all = set(['ShiftRoadFilter', 'ShiftLanesFilter',
                              'TiltRoadFilter', 'LaneWidthFilter',
                              'ConstantColorFilter', 'RandomColorMeanFilter'])
    
        self.op_params = params
        
    def get_filters(
        self,
        filters_to_output=['TiltRoadFilter'],
    ):
        for fil in filters_to_output:
            try:
                assert(fil in self.filter_set_all)
            except AssertionError:
                print('Your filter is not a valid filter \n')
                print('Please check the filter name again \n')
                raise
            else:
                pass
                
        filters_op_set = set(filters_to_output) 
        is_keep_layer = [ [True] * len(self.params['layers'][i]['filters']) 
                        for i in range(len(self.params['layers'])) ]
        copy_layer = deepcopy(self.params['layers'])

        for idx_layer, layer in enumerate(self.params['layers']):
            for idx_fil, fil_type in enumerate(layer['filters']):
                filter_type = fil_type['type']
        
                print(filter_type)
                if filter_type not in filters_op_set:
                    is_keep_layer[idx_layer][idx_fil] = False
        print(is_keep_layer)
        del_layer_count = 0
        for idx_layer, mask in enumerate(is_keep_layer):
            if sum(mask) is 0:

                del copy_layer[idx_layer - del_layer_count]
                del_layer_count += 1
                continue
            else:
                filters = []
                for idx_fil, is_filter in enumerate(mask):
                    if is_filter is True:
                        filters.append(copy_layer[idx_layer - del_layer_count]['filters'][idx_fil])
                copy_layer[idx_layer - del_layer_count]['filters'] = filters

        self.params['layers'] = copy_layer
        return self.params['layers']
    
    def get_layers(
        self,
        layer_types_list=['StraightRoadLayer'],
    ):
        for layer in layer_types_list:
            try:
                assert(layer in self.layer_set_all)
            except AssertionError:
                print('Your layer is not a valid filter \n')
                print('Please check the filter name again \n')
                raise
            else:
                pass
        
        layer_types_set = set(layer_types_list)
        copy_layer_config = deepcopy(self.params['layer_configs'])

        keys = list(self.params['layer_configs'])

        for idx, (conf, key) in enumerate(zip(self.params['layer_configs'], keys)):
            lyr_type = self.params['layer_configs'][conf]['layer_type']
            
            if lyr_type not in layer_types_set:
                del copy_layer_config[key]
                
        self.params['layer_configs'] = copy_layer_config

        return self.params
    
if __name__ == '__main__':
    params = sim_config()
    conf_manager = ConfigManager(params['simulator'])
    conf_manager.get_filters(['RandomColorMeanFilter', 'ConstantColorFilter', 'TiltRoadFilter' ])
    conf_manager.get_layers(['BackgroundLayer', 'SkyLayer', 'StraightRoadLayer' ])