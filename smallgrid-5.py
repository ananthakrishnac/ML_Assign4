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
import hiivemdptoolbox.hiive.mdptoolbox.mdp as mdp
from random import shuffle, randrange

def transition_matrix(S, A, prob, s_obstacle, s_terminal):
    transitions = {}
    rows, cols = S.shape
    P = np.zeros((len(A), rows * cols, rows * cols))
    #P = np.ones((len(A), rows * cols, rows * cols))

    for num in range(1, rows * cols + 1):
        div, mode = num//cols, num%cols
        if mode == 0:
            row, col = div-1, cols-1
        else:
            row, col = div, mode-1

        transitions[num] = {}
        transitions[num]['Stay'] = [row, col]

        if row != 0:
            if S[row-1,col] not in s_obstacle: 
                transitions[num]['Up'] = [row-1, col]
            else: 
                transitions[num]['Up'] = [row, col]
        else: 
            transitions[num]['Up'] = [row, col]

        if row != rows-1:
            if S[row+1,col] not in s_obstacle: 
                transitions[num]['Down'] = [row+1, col]
            else: 
                transitions[num]['Down'] = [row, col]
        else: 
            transitions[num]['Down'] = [row, col]

        if col != 0:
            if S[row,col-1] not in s_obstacle: 
                transitions[num]['Left'] = [row, col-1]
            else: 
                transitions[num]['Left'] = [row, col]
        else: 
            transitions[num]['Left'] = [row, col]

        if col != cols-1:
            if S[row,col+1] not in s_obstacle: 
                transitions[num]['Right'] = [row, col+1]
            else: 
                transitions[num]['Right'] = [row, col]
        else: 
            transitions[num]['Right'] = [row, col]

    for s in range(1,rows*cols+1):
        for index, direction in zip(range(4), ['Up','Down','Left','Right']):
            P[index, s-1, S[transitions[s][direction][0],transitions[s][direction][1]]-1] += 1-2*prob
            
        for index in range(4):
            for direction in ['Up','Down','Left','Right']:
                if index in (0, 1) and direction in ('Right', 'Left'):
                    P[index,s-1,S[transitions[s][direction][0],transitions[s][direction][1]]-1] += prob
                if index in [2, 3] and direction in ['Up', 'Down']:
                    P[index,s-1,S[transitions[s][direction][0],transitions[s][direction][1]]-1] += prob
        
    for term in s_terminal:
        P[0, term-1, :] = 0
        P[1, term-1, :] = 0
        P[2, term-1, :] = 0
        P[3, term-1, :] = 0
        P[0, term-1, term-1] = 1
        P[1, term-1, term-1] = 1
        P[2, term-1, term-1] = 1
        P[3, term-1, term-1] = 1

    return P, transitions

def reward_matrix(S, A, r, s_goal, r_goal, s_penalty, r_penalty, transitions, s_terminal):
    rows, cols = S.shape
    R = r*np.ones((rows * cols, len(A)))   
    landings = {'Up':0, 'Down':1, 'Left':2, 'Right':3}
    
    for s in range(1, rows*cols+1):
        for direction, index in landings.items():
            if S[transitions[s][direction][0],transitions[s][direction][1]] in s_goal:
                R[s-1, index] = r_goal[s_goal.index(S[transitions[s][direction][0],transitions[s][direction][1]])]
            if S[transitions[s][direction][0],transitions[s][direction][1]] in s_penalty:
                R[s-1, index] = r_penalty[s_penalty.index(S[transitions[s][direction][0],transitions[s][direction][1]])]
    
    for s in s_terminal: 
        R[s-1,:] = 0

    return R

def policy_mapping(policy):
    policy1 = policy.copy().astype(np.int)
    policy2 = policy.copy().astype(np.str)

    policy1[policy1==0] = 0
    policy1[policy1==1] = 0
    policy1[policy1==2] = 0
    policy1[policy1==3] = 0
    policy1[policy1==4] = 1
    policy1[policy1==5] = 2
    policy1[policy1==6] = 3

    policy2[policy2=='0'] = '↑'
    policy2[policy2=='1'] = '↓'
    policy2[policy2=='2'] = '←'
    policy2[policy2=='3'] = '→'
    policy2[policy2=='4'] = 'B'
    policy2[policy2=='5'] = 'G'
    policy2[policy2=='6'] = 'A'

    return policy1, policy2

def draw_gridworld(utility_list, policy_list, iteration_list, name_list, figname, row, col):
    plt.close()
    plt.cla()
    plt.clf()
    item_count = len(policy_list)
    fig, ax = plt.subplots(2, item_count, figsize=(16.5,8.5))
    
    for index, utility in enumerate(utility_list):
        ax1 = sns.heatmap(utility, cmap= 'YlOrBr',  #cmap="Blues",
                    linewidths=0.5,
                    linecolor='black',
                    cbar=False,
                    square=True,
                    mask=(utility==1e-10),
                    ax=ax[0,index])

    palette = np.array([[211, 211, 211], [  0,   0,   0], [  0, 100,   0], [255,   0,   0]])
    colors, texts = [], []
    for policy in policy_list:
        policy1, policy2 = policy_mapping(policy)
        colors.append(policy1)
        texts.append(policy2)

    for index, color in enumerate(colors):
        ax[1, index].imshow(palette[color].astype(np.uint8))
    
    for i in range(row):
        for j in range(col):
            for k in range(item_count):
                tmp = texts[k]
                text = ax[1, k].text(j, i, tmp[i, j], ha="center", va="center", color="black")
            
    for i in range(item_count):        
        ax[0, i].set_facecolor("black")
        ax[0, i].set_title('{} - {} Iterations'.format(name_list[i], iteration_list[i]), fontsize=13)
        ax[1, i].set_facecolor("white")
        ax[1, i].set_title('{} - {} Iterations'.format(name_list[i], iteration_list[i]), fontsize=13)

        ax[0, i].set_xticks(np.arange(col))
        ax[0, i].set_yticks(np.arange(row))
        ax[0, i].set_xticklabels(list(range(col)))
        ax[0, i].set_yticklabels(list(range(row))[::-1])
        
        ax[1, i].set_xticks(np.arange(col))
        ax[1, i].set_yticks(np.arange(row))
        ax[1, i].set_xticklabels(list(range(col)))
        ax[1, i].set_yticklabels(list(range(row))[::-1])

    plt.tight_layout()
    #plt.show()
    plt.savefig(figname, format='png', dpi=150,bbox_inches='tight')
    #


def draw_iterVsTime(vi_iterations, vi_runtimes, pi_iterations, pi_runtimes, figname="10.Iter-Times.png"):
    plt.close()
    
    plt.plot(vi_iterations, vi_runtimes, pi_iterations, pi_runtimes)
    plt.xlabel("Iteration")
    plt.ylabel("Time")
    plt.legend(['Value Iteration','Policy Iteration'])
    
    plt.savefig(figname, format='png', dpi=150,bbox_inches='tight')



def main():
    row, col = 25, 25                # Number of rows & cols in grid world
    s_obstacle = [
        13,14,16,17,18,19,20,21,22,23,24,25,
        26,27,28,29,30,31,32,33,34,35,36,38,39,42,46,47,48,49,50,
        51,63,64,65,69,75,
        76,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,100,
        101,103,109,110,111,112,113,114,125,
        126,128,130,132,139,141,142,143,144,145,146,147,148,149,150,
        151,153,155,157,159,160,161,162,164,166,175,
        176,178,180,182,184,189,191,193,194,195,196,197,198,200,
        201,203,205,207,209,211,212,213,214,216,218,222,223,225,
        226,228,230,231,232,234,235,236,237,238,239,241,243,245,247,248,250,
        251,253,257,259,263,264,266,268,270,272,273,275,
        276,278,280,282,284,286,288,289,291,293,295,297,298,300,
        301,303,305,307,309,311,316,318,320,322,323,325,
        326,328,330,332,334,336,337,338,339,340,341,343,345,347,348,350,
        351,353,355,357,359,361,362,364,366,368,370,372,373,375,
        376,378,380,381,382,384,389,391,393,395,397,398,400,
        401,403,405,407,409,410,411,412,414,416,418,420,422,423,425,
        426,428,432,434,435,436,437,439,441,443,445,447,448,450,
        451,453,455,457,464,466,468,470,472,473,475,
        476,478,480,481,482,483,484,485,486,487,488,489,491,493,495,497,498,500,
        501,503,518,520,522,523,525,
        526,528,530,531,532,533,534,535,536,537,538,539,540,541,542,543,545,547,548,550,
        551,553,561,563,565,567,568,570,573,575,
        576,578,579,580,581,582,583,584,586,588,590,592,593,595,596,597,598,600,
        601,625
        ]     # Location of obstacles, (0,0) = starts at 1, count row wise horizontally
    s_terminal = [15]               # Goal to reach
    s_goal = [15]                   # Goal value
    r_goal = [1000]                   # 
    s_penalty = [206,356,406,456,560,572,585]                # Goal to avoid - penalty block
    r_penalty = [-200,-200,-200,-200,-200,-200,-200]               # Penalty points
    
    S = (np.arange(row*col)+1).reshape(row, col)
    A = ['Up','Down','Left','Right']
    prob = 0.1
    epsilon = 0.001
    gamma = 0.99    # Performs better at 250. 0.9 Does not perform as good and stops at 129
    r = -0.1
    P, transitions = transition_matrix(S, A, prob, s_obstacle, s_terminal)
    R = reward_matrix(S, A, r, s_goal, r_goal, s_penalty, r_penalty, transitions, s_terminal)
    
    
    max_iterations=500000
    params_iterations(P, R, max_iterations,"Maze")

    
    vi_utilitys, vi_policys, vi_iterations, vi_runtimes = run_vi(P, R, gamma, 
                                                                  [50,100,175,250],
                                                                  epsilon,row,col,
                                                                  s_obstacle,s_terminal,s_penalty)
    
    draw_gridworld(vi_utilitys, vi_policys, vi_iterations, ['VI','VI','VI','VI'], '1.maze-vi-1.png',row, col)
    
    print("Value Iterations - Maze")
    for index, utility in enumerate(vi_utilitys):
        print(index,  np.amax(utility), vi_iterations[index])
    
    
    pi_utilitys, pi_policys, pi_iterations, pi_runtimes = run_pi(P, R, gamma,
                                                                  [5,20,50,75], 
                                                                  epsilon,row,col,
                                                                  s_obstacle,s_terminal,s_penalty)
    
    draw_gridworld(pi_utilitys, pi_policys, pi_iterations, ['PI','PI','PI','PI'], '2.maze-pi-1.png',row, col)
    
    print("Policy Iterations - Maze")
    for index, utility in enumerate(pi_utilitys):
        print(index,  np.amax(pi_utilitys), pi_iterations[index])
    

    draw_iterVsTime(vi_iterations, vi_runtimes, pi_iterations, pi_runtimes)

    
    q_utilitys, q_policys, iteration, q_runtimes = run_qlearning(P, R, gamma, 
                                                                  [10000,50000,100000,500000], 
                                                                  #[100,1000,5000,10000], 
                                                                  row, col,
                                                                  s_obstacle,s_terminal,s_penalty)
    
    draw_gridworld(q_utilitys, q_policys, iteration, ['Q','Q','Q','Q'], '3.maze-q-p1-1.png',row, col)
    
    print("Q learning Iterations - Maze")
    for index, utility in enumerate(pi_utilitys):
        print(index,  np.amax(q_utilitys), iteration[index])
    
    #draw_iterVsTime(iteration, q_runtimes, pi_iterations, pi_runtimes, figname="Qlearning times.png")
    
    # TODO: PLOT iterations (x-axis) vs Runtime (Y-axis)
    
if __name__ == '__main__':
    main()