# function to generate paltas config file for a doppelganger
# requires a pickled .txt file of forward model outputs

import json
import pickle
import numpy as np
from paltas.MainDeflector.simple_deflectors import PEMDShear

#TODO: create helper functions for each config component

def main_deflector_dict(kwargs_result,z_lens):
    kwargs_PEMD = kwargs_result['kwargs_lens'][0]
    kwargs_shear = kwargs_result['kwargs_lens'][1]
    dict = {
        'class': PEMDShear,
        'parameters':{
            'z_lens':z_lens,
            'gamma':kwargs_PEMD['gamma'],
            'theta_E':kwargs_PEMD['theta_E'],
            'e1':kwargs_PEMD['e1'],
            'e2':kwargs_PEMD['e2'],
            'center_x':kwargs_PEMD['center_x'],
            'center_y':kwargs_PEMD['center_y'],
            'gamma1':kwargs_shear['gamma1'],
            'gamma2':kwargs_shear['gamma2'],
            'ra_0':kwargs_shear['ra_0'],
            'dec_0':kwargs_shear['dec_0']
        }
    }
    
    return dict


def doppelganger_config(input_file,output_path,lens_catalog_row):
    """
    Args: 
        input_file: location of input file
        output_path: where to write .py config file
        lens_catalog_row (pandas Series): Single row of lens system from lens catalog
    """
    
    # load in variables from input file
    f = open(input_file,'rb')
    (multi_band_list, kwargs_model, kwargs_result,
            image_likelihood_mask_list) = pickle.load(f)
    f.close()
    
    # load in catalog file
    
    print(multi_band_list)
    
    
    test_dict = {
        'param1':100,
        'param2':200
    }

    with open(output_path,"w") as dict_file:
        json.dump(test_dict,dict_file,indent=1)

