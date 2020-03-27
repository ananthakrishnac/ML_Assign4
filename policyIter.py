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


def run_pi(P, R, gamma=0.99, iteration=[10,100,1000,10000], epsilon=0.01, row=0, col=0, s_obstacle=None, s_terminal=None, s_penalty=None, figurename=None):
    pi_utilitys = []
    pi_policys = []
    pi_iterations = []
    pi_runtimes = []
    for i in iteration:
        utility, policy, iteration, time1 = policy_iteration(P, R, gamma, i, epsilon, row, col, s_obstacle, s_terminal, s_penalty)
        pi_utilitys.append(utility)
        pi_policys.append(policy)
        pi_iterations.append(iteration)
        pi_runtimes.append(time1)
    print(pi_runtimes)
    
    return pi_utilitys, pi_policys, pi_iterations, pi_runtimes

def policy_iteration(P, R, gamma, iteration, epsilon, row=0, col=0, s_obstacle=None, s_terminal=None, s_penalty=None):
    #print("-->"+str(iteration))
    pi = mdp.PolicyIteration(transitions = P, 
                                  reward = R, 
                                  gamma = gamma, 
                                  max_iter = iteration 
                                , skip_check=True       # Avoid checking Transition and Reward for now.
                                )
    # pi = mdp.PolicyIterationModified(transitions = P, 
    #                          reward = R, 
    #                          gamma = gamma,
    #                          epsilon = epsilon,
    #                          max_iter = iteration 
    #                         , skip_check=True       # Avoid checking Transition and Reward for now.
    #                         )
    pi.max_iter = iteration
    pi.discount = gamma
    pi.run()
    utility = pi.V
    policy = pi.policy
    iterations = pi.iter
    run_time = pi.time
    
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
    
    #print(iterations)
    
    return utility, policy, iterations, run_time