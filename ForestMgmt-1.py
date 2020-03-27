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
        print(index, color)
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
    print(P)
    print(R)
    
    q_utilitys, q_policys, iteration, q_runtimes = run_qlearning(P, R, gamma, [50000,50000000], row, col)
    draw_gridworld(q_utilitys, q_policys, iteration, ['Q','Q','Q','Q'], 'q-5B-forest.png',row, col)
    
if __name__ == '__main__':
    main()