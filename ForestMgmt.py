from valueIter import value_iteration, run_vi
from policyIter import policy_iteration, run_pi
from qlearner import q_learning, run_qlearning
from iterPlots import params_iterations
import time
import copy
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random
import mdptoolbox.example as mdpTBEx
import hiivemdptoolbox.hiive.mdptoolbox.mdp as mdp
from random import shuffle, randrange


def policy_mapping(policy):
    policy1 = policy.copy().astype(np.int)
    policy2 = policy.copy().astype(np.str)

    policy1[policy1==0] = 0
    policy1[policy1==1] = 1

    policy2[policy2=='0'] = '.'
    policy2[policy2=='1'] = '↓'
    
    return policy1, policy2

def draw_gridworld(utility_list, policy_list, iteration_list, name_list, figname, row, col):
    plt.close()
    plt.cla()
    plt.clf()
    item_count = len(policy_list)
    fig, ax = plt.subplots(1, item_count, figsize=(16.5,8.5))
    
    palette = np.array([[179, 228, 36], [  255,   234,   211], [  0, 100,   0], [255,   0,   0]])
    colors, texts = [], []
    for policy in policy_list:
        policy1, policy2 = policy_mapping(policy)
        colors.append(policy1)
        texts.append(policy2)

    
    for index, color in enumerate(colors):
        #print(index, color)
        c = palette[color].astype(np.uint8)
        a = ax[index]
        a.imshow(c)
    
    for i in range(row):
        for j in range(col):
            for k in range(item_count):
                tmp = texts[k]
                text = ax[k].text(j, i, tmp[i, j], ha="center", va="center", color="black")
            
    for i in range(item_count):        
        ax[i].set_facecolor("white")
        ax[i].set_title('{} - {} Iterations'.format(name_list[i], iteration_list[i]), fontsize=13)
        
        ax[i].set_xticks(np.arange(col))
        ax[i].set_yticks(np.arange(row))
        ax[i].set_xticklabels(list(range(col)))
        ax[i].set_yticklabels(list(range(row))[::-1])

    plt.tight_layout()
    #plt.show()
    plt.savefig(figname, format='png', dpi=150,bbox_inches='tight')

def main():        
    row, col = 50, 50 
    s_terminal = [0]               # Goal to reach
    s_goal = [0]                   # Goal value
    r_goal = [0]                   # 
    s_penalty = [0]                # Goal to avoid - penalty block
    r_penalty = [0]               # Penalty points
    prob = 0.1
    #epsilon = 0.01
    epsilon = 0.01
    #gamma = 0.9    # Does not perform as good and stops at 129
    gamma = 0.99    # Performs better at 250
    r = -0.1
    #r = -0.04
    S=2500
    #S=1000
    r1=400
    r2=1
    p=.1
    A = 2
    max_iterations=5000000
    #Set the environment
    #np.random.seed(1729)
    P, R = mdpTBEx.forest(S=S, r1=r1, r2=r2, p=p, is_sparse=False)
    #print(P)
    #print(R)
    
    params_iterations(P, R, max_iterations, "Forest")
    
    vi_utilitys, vi_policys, vi_iterations, vi_runtimes = run_vi(P, R, gamma, 
                                                                  [50,100,175,250],
                                                                  epsilon,row,col)
    
    draw_gridworld(vi_utilitys, vi_policys, vi_iterations, ['VI','VI','VI','VI'], '1.forest-vi-1.png',row, col)
    
        
    print("Value Iterations - Forest")
    for index, utility in enumerate(vi_utilitys):
        print(index,  np.amax(utility), vi_iterations[index])
    
    
    pi_utilitys, pi_policys, pi_iterations, pi_runtimes = run_pi(P, R, gamma, [5,20,50,75], epsilon, row, col)
    draw_gridworld(pi_utilitys, pi_policys, pi_iterations, ['PI','PI','PI','PI'], '2.forest-pi-1.png',row, col)
    
    print("Policy Iterations - Forest")
    for index, utility in enumerate(pi_utilitys):
        print(index,  np.amax(pi_utilitys), pi_iterations[index])

    q_utilitys, q_policys, iteration, q_runtimes = run_qlearning(P, R, gamma, [10000,50000,100000,500000], row, col)
    #q_utilitys, q_policys, iteration, q_runtimes = run_qlearning(P, R, gamma, [100,100,100,100], row, col)
    draw_gridworld(q_utilitys, q_policys, iteration, ['Q','Q','Q','Q'], '3.forest-q-1.png',row, col)
    
    print("Q learning Iterations - Forest")
    for index, utility in enumerate(pi_utilitys):
        print(index,  np.amax(q_utilitys), iteration[index])
    
if __name__ == '__main__':
    main()