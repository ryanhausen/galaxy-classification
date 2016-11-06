# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 15:30:41 2016

@author: ryan
"""

import commands
import multiprocessing


# async process
def _run_net(param_tuple):
    # gpu_ids: string with comma seperated gpu ids
    # main_params: single string of terminal args
    gpu_ids, main_params = param_tuple
    
    command_string = 'CUDA_VISIBLE_DEVICES={} python main.py {}'
    command_string = command_string.format(gpu_ids, main_params)
    
    stat, out = commands.getstatusoutput(command_string)
    
    return (gpu_ids, stat)


def _get_default_params():
    None
    
def _get_range_params():
    None

def _param_search(gpus):
    default_params = _get_default_params()
    range_params = _get_range_params()
    best_params = {k:None for k in range_params.keys()}    
    
    jobs = {gpu:None for gpu in gpus}

    for param, vals in range_params.items():
        



if __name__ == "__main__":
    # parse GPU ids
    None