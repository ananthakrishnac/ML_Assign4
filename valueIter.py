import time
import copy
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random
#import hiive.mdptoolbox as mdp #import mdp
import hiivemdptoolbox.hiive.mdptoolbox.mdp as mdp
from random import shuffle, randrange

def value_iteration(P, R, gamma, iteration, epsilon, row=0, col=0, s_obstacle=None, s_terminal=None, s_penalty=None):

    vi = mdp.ValueIteration(transitions = P, 
                                reward = R, 
                                gamma = gamma, 
                                max_iter = iteration, 
                                epsilon = epsilon,
                                initial_value = 0 
                                , skip_check=True # Enable checking input data for now to save some time.
                                )
    
    vi.max_iter = iteration
    vi.discount = gamma
    vi.run()
    policy = vi.policy
    utility = vi.V
    iterations = vi.iter
    run_time = vi.time
    
    policy = list(copy.copy(policy))
    utility = list(copy.copy(utility))
    
    if s_obstacle != None:
        for s in s_obstacle:
            policy[s-1] = 4
            utility[s-1] = 1e-10

    if s_terminal != None:        
        for s in s_terminal:
            policy[s-1] = 5
            utility[s-1] = 1e-10
    
    if s_penalty != None:
        for s in s_penalty:
            policy[s-1] = 6
            utility[s-1] = 1e-10
            
    policy = np.array(policy).reshape(row, col)
    utility = np.array(utility).reshape(row, col)

    return utility, policy, iterations, run_time


def run_vi(P, R, gamma=0.99, iteration=[10,100,1000,10000], epsilon=0.01, row=0, col=0, s_obstacle=None, s_terminal=None, s_penalty=None, figurename=None):
    vi_utilitys = []
    vi_policys = []
    vi_iterations = []
    vi_runtimes = []
    for i in iteration:
        #utility, policy, iteration, time1 = value_iteration(P, R, gamma, i, epsilon, row, col)
        utility, policy, iteration, time1 = value_iteration(P, R, gamma, i, epsilon, row, col, s_obstacle, s_terminal, s_penalty)
        
        vi_utilitys.append(utility)
        vi_policys.append(policy)
        vi_iterations.append(iteration)
        vi_runtimes.append(time1)
    print(vi_runtimes)
    
    return vi_utilitys, vi_policys, vi_iterations, vi_runtimes
