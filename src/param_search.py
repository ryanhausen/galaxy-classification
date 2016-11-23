# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 15:30:41 2016

@author: ryan
"""
import os
import subprocess as sp
import multiprocessing mp

from copy import deepcopy


# async process
def _run_net(param_tuple):
    # gpu_ids: string with comma seperated gpu ids
    # main_params: single string of terminal args
    gpu_ids, main_params = param_tuple
    
    tmp_env = os.environ.copy()
    tmp_env['CUDA_VISIBLE_DEVICES'] = gpu_ids
    
    cmd = ['python', 'main.py'] + main_params    
    
    
    command_string = 'python main.py {}'
    command_string = command_string.format(gpu_ids, main_params)
    
    #stat, out = commands.getstatusoutput(command_string)
    
    return commands.getstatusoutput(command_string)

def _dict_to_cmd(dictionary):
    val = ''
    
    for k, v in dictionary.items():
        val += '-{} {} '.format(k,v)
        
    return val

def _get_default_params():
    None
    
def _get_range_params():
    None

def _param_search(gpus):
    default_params = _get_default_params()
    range_params = _get_range_params()
    best_params = {k:None for k in range_params.keys()}    
    
    jobs = {gpu:None for gpu in gpus}
    
    for param in range_params.keys():
        print 'Evaluating: {}'.format(param)        
        
        vals = deepcopy(range_params[param])
        best = None
        
        
        
        
        
        
    
            
        
        
    
    
    
        
    
        



if __name__ == "__main__":
    # parse GPU ids
    None