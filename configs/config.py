"""
Configuration for image simulator.

@author: Sebastian Lotter <sebastian.g.lotter@fau.de>
"""
def sim_config():
    """
    Example simulator configuration.
    """
    return {
    'simulator': {
        'height': 256,
        'width' : 256,
        'layers': [
            {
                'id': 'Back0',
                'config_id': '0',
                'filters' : [
                    {
                        'type'   : 'RandomColorMeanFilter',
                        'params' : {
                            'dr' : 100,
                            'dg' : 100,
                            'db' : 100
                        }
                    }
                ]
            },
            {
                'id': 'Sky0',
                'config_id': '1',
                'prob' : 1,  #probability for layer appearing in image
                'filters' : [
                    {
                        'type'   : 'ConstantColorFilter',
                        'params' : {
                            'dr' : 100,
                            'dg' : 100,
                            'db' : 100
                        }
                    }
                ]
            },
            {
                'id': 'Road0',
                'config_id': '2',
                'prob' : 1, #probability for layer appearing in image
                'serialize': 1,
                'filters' : [
                    {
                        'type'   : 'ShiftRoadFilter',
                        'params' : {
                            'lb' : -0.2,
                            'ub' :  0.2
                        }
                    },
                    {
                        'type'   : 'ShiftLanesFilter',
                        'params' : {
                            'lb' : -0.02,
                            'ub' :  0.02
                        }
                    },
                    {
                        'type'   : 'TiltRoadFilter',
                        'params' : {
                            'lb' : -1.5,
                            'ub' :  1.5
                        }
                    },
                    {
                        'type'   : 'LaneWidthFilter',
                        'params' : {
                            'lb' : -0.010,
                            'ub' :  0.010
                        }
                    }
                ]
            }
        ],
        'layer_configs': {
            '0': {
                'layer_type'  : 'BackgroundLayer',
                'layer_params': {
                    'color_fct': {
                        'type': 'random',
                        'params': {
                            'mean'    : [0,120,80],
                            'range'   : 20
                        }   
                    }
                }
            },
            '1' : {
                'layer_type': 'SkyLayer',
                'layer_params': {
                    'color_fct': {
                        'type': 'constant',
                        'params': {
                            'color': [0,180,180]
                        }   
                    },
                    'shape': [[0.0,0.1], [0.0,0.7], [0.3,0.5], [0.0,0.1]]
                }
            },
            '2' : {
                'layer_type': 'StraightRoadLayer',
                'layer_params': {
                    'road': [
                        [1.0,0.3],
                        [0.0,0.3],
                    ],
                    'road_width': 0.4,
                    'lanes': [
                        [[1.0,0.3], [0.0,0.3]],
                        [[1.0,0.48], [0.0,0.48]],
                        [[1.0,0.51], [0.0,0.51]],
                        [[1.0,0.69], [0.0,0.69]]
                    ],
                    'lane_widths': [
                        0.01, 0.01, 0.01, 0.01
                    ],
                    'tilt': 0,
                    "transform_coordinates": {
                        "src": [
                            [0.0, 0.3], [0.0, 0.7], [1.0, 0.7], [1.0, 0.3]
                        ],
                        "tgt": [
                            [0.3, 0.45], [0.3, 0.55], [1.0, 1.0], [1.0, 0.0]
                        ]
                    },
                    'color_fcts': [
                        {
                            'type': 'random',
                            'params': {
                                'mean': [80,80,80],
                                'range'   : 10
                            }   
                        },
                        {
                            'type': 'constant_random_mean',
                            'params': {
                                'mean': [229,226,52],
                                'lb'  : -100,
                                'ub'  : 100
                            }   
                        },
                        {
                            'type': 'constant_random_mean',
                            'params': {
                                'mean': [229,226,52],
                                'lb'  : -100,
                                'ub'  : 100
                            }   
                        },
                        {
                            'type': 'constant_random_mean',
                            'params': {
                                'mean': [229,226,52],
                                'lb'  : -100,
                                'ub'  : 100
                            }   
                        },
                        {
                            'type': 'constant_random_mean',
                            'params': {
                                'mean': [229,226,52],
                                'lb'  : -100,
                                'ub'  : 100
                            }   
                        }
                    ]
                    
                }
            },
            
        }
    }
}

