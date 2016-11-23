# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 15:30:41 2016

@author: ryan
"""
import os
import subprocess as sp
import multiprocessing as mp
import json


from time import sleep
from copy import deepcopy


# async process
def _run_net(gpu_ids, main_params, result):
    # gpu_ids: string with comma seperated gpu ids
    # main_params: single string of terminal args
    #gpu_ids, main_params, result = param_tuple
    
    tmp_env = os.environ.copy()
    tmp_env['CUDA_VISIBLE_DEVICES'] = str(gpu_ids)
    
    cmd = ['python', 'main.py'] + _dict_to_cmd(main_params)
    
    print cmd    
    
    proc = sp.Popen(cmd, env=tmp_env, stdout=sp.PIPE)
    
    proc    
    
    result[gpu_ids] =  float(proc.communicate()[0].strip())

def _dict_to_cmd(dictionary):
    cmd = []
    for k, v in dictionary.items():    
        cmd.append('-{}'.format(k))
        cmd.append(str(v))
    return cmd

# this builds the directories for storing the results from each run
def _create_param_env(dictionary, run_num):
    report_dir = '../report/run_{}'.format(run_num)
    model_dir = '../models/run_{}'.format(run_num)
    
    os.mkdir(report_dir)
    os.mkdir(model_dir)
    
    dictionary['train_progress'] = os.path.join(report_dir, 'train_progress.csv')
    dictionary['test_progress'] = os.path.join(report_dir, 'test_progress.csv')
    dictionary['model_dir'] = model_dir
    
    return dictionary

def _get_default_params():
    with open('params.json', 'r') as f:
        return json.load(f)
    
def _get_range_params():
    with open('param_search.json', 'r') as f:
        return json.load(f)

def _param_search(gpus):
    default_params = _get_default_params()
    range_params = _get_range_params()
    best_params = {k:None for k in range_params.keys()}    
    
    # holds gpus and there processes
    jobs = {gpu:(None,None) for gpu in gpus}
    
    # holds the results of a processes run
    results = {gpu:None for gpu in gpus}

    # set up a directory for each run to save their results
    run_num = 0
    # use random descent for parameter search 
    #TODO switch to a gaussian process?  
    for param in range_params.keys():
        print 'Evaluating: {}'.format(param)        
        
        vals = deepcopy(range_params[param])
        # the current eval function is the top 1 accuracy so 0.0 is the poorest value
        best = 0.0
        
        # we're using a stack so we'll pop values as we use them
        while len(vals) > 0:
            print 'checking for ready GPUs'
            # check to see if any of the processes are done and run next config
            for gpu, job in jobs.items():
                proc, tested_val = job
                # first time start                
                if proc is None:
                    print 'starting process for the first time'
                    run_num += 1
                    next_val = vals.pop()
                    default_params[param] = next_val
                    default_params = _create_param_env(default_params, run_num)
                    param_tuple = (gpu, default_params, results)
                    jobs[gpu] = (mp.Process(target=_run_net, args=param_tuple), next_val)
                    jobs[gpu][0].start()
                # processed has finished its work
                elif proc.is_alive() == False:
                    print 'process is alive: {}'.format(proc.is_alive())                    
                    
                    if results[gpu] > best:
                        best = results[gpu]
                        best_params[param] = (tested_val, best)

                    run_num += 1
                    next_val = vals.pop()
                    default_params[param] = next_val
                    default_params = _create_param_env(default_params, run_num)
                    
                    param_tuple = (gpu, default_params, results)
                    
                    jobs[gpu] = (mp.Process(target=_run_net, args=param_tuple), next_val)
                    jobs[gpu][0].start()

            wait_time = 2*60
            print 'Waiting {} seconds'.format(wait_time)
            # None are finished, so wait and check again
            sleep(wait_time)

        #save the best val, and then move on to the next param
        default_params[param] = best_params[param]

    # save the best params
    with open('./best_params.json', 'w') as f:
        json.dump(best_params, f)

if __name__ == "__main__":
    # pass a list of GPU ids
    _param_search([0])