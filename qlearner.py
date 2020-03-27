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


def run_qlearning(P, R, gamma=0.99, iteration=[10,100,1000,10000], row=0, col=0, s_obstacle=None, s_terminal=None, s_penalty=None, figurename=None):
    q_matrixs = []
    q_utilitys = []
    q_policys = []
    q_discrepancy = []
    q_runtimes = []
    for i in iteration:
        start = time.time()
        matrix, utility, policy, discrepancy = q_learning(P, R, gamma, i, row, col, s_obstacle, s_terminal, s_penalty)
        #matrix, utility, policy, discrepancy = q_learning(P, R, gamma, i)
        end = time.time()
        q_runtimes.append(end - start)
        q_matrixs.append(matrix)
        q_utilitys.append(utility)
        q_policys.append(policy)
        q_discrepancy.append(discrepancy)
    print(q_runtimes)
    
    return q_utilitys, q_policys, iteration, q_runtimes
    #draw_gridworld(q_utilitys, q_policys, iteration, ['Q','Q','Q','Q'], figurename,row, col)


def q_learning(P, R, gamma, iteration, row=0, col=0, s_obstacle=None, s_terminal=None, s_penalty=None):
    ql = mdp.QLearning(transitions = P, 
                           reward = R, 
                           gamma = gamma, 
                           n_iter = iteration 
                           , skip_check=True # Skip check for now to go over checking Transition and Reward matrix
                           )
    ql.max_iter = iteration
    ql.run()
    q_matrix = ql.Q
    utility = ql.V
    policy = ql.policy
    mean_discrepancy = ql.error_mean
    
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
    
    return q_matrix, utility, policy, mean_discrepancy
