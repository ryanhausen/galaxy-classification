# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 15:30:41 2016

@author: ryan
"""
import os
import subprocess as sp
import multiprocessing as mp
import json

from colorama import init, Fore
init(autoreset=True)

from time import sleep
from copy import deepcopy


class capacity_list(object):
    """
        This is a helper class that holds values from the parameter search.
        It should store tuples like (paramval, score)
    """    
    def __init__(self, capacity):
        self.capacity = capacity
        self.items = []
        self.at_capacity = False
    
    def append(self, paramval, score):
        self.items.append((paramval, score))
        self.at_capacity = len(self.items) == self.capacity

    
    def get_best_val(self):
        best_param = None
        best_score = 0.0
        
        for pval, score in self.items:
            if score > best_score:
                best_param = pval
                best_score = score
                
        return best_param

# async process
def _run_net(gpu_ids, main_params, result):
    # gpu_ids: string with comma seperated gpu ids
    # main_params: single string of terminal args
    #gpu_ids, main_params, result = param_tuple
    
    tmp_env = os.environ.copy()
    tmp_env['CUDA_VISIBLE_DEVICES'] = str(gpu_ids)
    
    cmd = ['python', 'main.py'] + _dict_to_cmd(main_params)
    
    #print cmd    
    
    proc = sp.Popen(cmd, env=tmp_env, stdout=sp.PIPE)
    
    result[gpu_ids] = float(proc.communicate()[0].strip())

def _dict_to_cmd(dictionary):
    cmd = []
    for k, v in dictionary.items():    
        cmd.append('-{}'.format(k))
        
        if type(v) != list:        
            cmd.append(str(v))
        else:
            p = '['
            
            for i in v:
                if type(i) == str:
                    p += i + ','
                else:
                    p += str(i) + ','
                
            p = p[:-1] + ']'
            
            cmd.append(p)            
            
    return cmd

def _log(msg):
    with open('./param_search_log', 'a') as f:
        f.write(msg + '\n')

# this builds the directories for storing the results from each run
def _create_param_env(dictionary, run_num):
    report_dir = '../report/run_{}'.format(run_num)
    model_dir = '../models/run_{}/'.format(run_num)
    
    os.mkdir(report_dir)
    os.mkdir(model_dir)
    
    dictionary['train_progress'] = os.path.join(report_dir, 'train_progress.csv')
    dictionary['test_progress'] = os.path.join(report_dir, 'test_progress.csv')
    dictionary['model_dir'] = model_dir
    
    with open(os.path.join(report_dir,'params.json'), 'w') as f:
        json.dump(dictionary, f)
    
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
    tmp_best_params = {k:capacity_list(len(range_params[k])) for k in range_params.keys()}
    
    
    # holds gpus and there processes
    jobs = {gpu:(None,None,None) for gpu in gpus}
    
    # holds the results of a processes run
    results = mp.Manager().dict()
    for gpu in gpus:
        results[gpu] = None

    # set up a directory for each run to save their results
    run_num = 0
    # use random descent for parameter search 
    #TODO switch to a gaussian process?  
    
    # the current eval function is the top 1 accuracy so 0.0 is the poorest value
    best = 0.0    
    
    for param in range_params.keys():
        msg = 'Evaluating: {}'.format(param)
        print Fore.CYAN + msg
        _log(msg)
        
        vals = deepcopy(range_params[param])
        
        # we're using a stack so we'll pop values as we use them
        while len(vals) > 0:
            msg = 'Checking for ready GPUs'
            print Fore.BLUE + msg
            _log(msg)
            
            # check to see if any of the processes are done and run next config
            for gpu, job in jobs.items():
                proc, tested_val, tested_param = job
                # first time start                
                if proc is None:
                    msg = 'Starting Job on GPU: {}'.format(gpu)
                    print Fore.MAGENTA + msg
                    _log(msg)                    
                    
                    run_num += 1
                    next_val = vals.pop()
                    default_params[param] = next_val
                    default_params = _create_param_env(default_params, run_num)
                    param_tuple = (gpu, default_params, results)
                    jobs[gpu] = (mp.Process(target=_run_net, args=param_tuple), next_val, param)
                    jobs[gpu][0].start()
                # processed has finished its work
                elif proc.is_alive() == False:
                    
                    msg = 'GPU: {} finished\nCurrent Best: {} Result: {}'.format(gpu, best, results[gpu])                    
                    print Fore.GREEN + msg
                    _log(msg)                    
                    
                    if results[gpu] > best:
                        best = results[gpu]
                        #best_params[param] = (tested_val, best)  

                    tmp_best_params[tested_param].append(tested_val, results[gpu])
                    
                    if tmp_best_params[tested_param].at_capacity:
                        best_params[tested_param] = tmp_best_params[tested_param].get_best_val()
                        default_params[tested_param] = best_params[tested_param]
                            
                    msg = 'Starting Job on GPU: {}'.format(gpu)
                    print Fore.MAGENTA + msg
                    _log(msg)                    

                    run_num += 1
                    next_val = vals.pop()
                    default_params[param] = next_val
                    default_params = _create_param_env(default_params, run_num)
                    
                    param_tuple = (gpu, default_params, results)
                    
                    jobs[gpu] = (mp.Process(target=_run_net, args=param_tuple), next_val)
                    jobs[gpu][0].start()

            wait_time = 2*60
            msg = 'Waiting {} seconds'.format(wait_time)
            print Fore.YELLOW + msg
            # None are finished, so wait and check again
            sleep(wait_time)

    # save the best params
    with open('./best_params.json', 'w') as f:
        json.dump(best_params, f)

if __name__ == "__main__":
    # pass a list of GPU ids
    _param_search([0])