from valueIter import value_iteration, run_vi
from policyIter import policy_iteration, run_pi
from qlearner import q_learning, run_qlearning
import time
import copy
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random
#import hiive.mdptoolbox as mdp #import mdp
import mdptoolbox.example as mdpTBEx
import hiivemdptoolbox.hiive.mdptoolbox.mdp as mdp
from random import shuffle, randrange

def params_discount_iterations(policy, P, R, max_iterations):
    print("Exploration In Discount")
    policies = []
    iterations = []
    times = []
    max_value = []

    gamma = np.arange(start=.001, stop=1, step=.010)
     
    for d in gamma:
        if policy==True:
            gamma_iterations = mdp.ValueIteration(P, R, d, max_iter=max_iterations)
        else:
            gamma_iterations = mdp.PolicyIterationModified(P, R, d,max_iter=max_iterations)
            #gamma_iterations = mdp.PolicyIteration(P, R, d,max_iter=max_iterations)
        
        gamma_iterations.run()
        #print("Discount Value: " +str(d) +" Iterations to Converge:", discount_iterations.iter)
        #print("Discount Value: " +str(d) +" Time:", discount_iterations.time)
        policies.append(gamma_iterations.policy)
        iterations.append(gamma_iterations.iter)
        times.append(gamma_iterations.time)
        max_value.append(np.amax(gamma_iterations.V))
        
    return policies,iterations,times,gamma, max_value


def params_epsilon_iterations(policy, P, R, max_iterations):
    print("Exploration with Epsilon")
    policies = []
    iterations = []
    times = []
    max_value = []
    
    #epsilon_values=np.arange(start=.001, stop=1, step=.005) 
    #epsilon_values=np.arange(start=.1, stop=100, step=0.5)
    #epsilon_values=list(np.arange(start=.001, stop=1, step=.005)) + list(np.arange(start=.1, stop=100, step=0.5))
    epsilon_values=list(np.arange(start=.001, stop=1, step=.005)) + list(np.arange(start=.1, stop=10, step=0.5))
    epsilon_values=np.array(epsilon_values)
    
    #print(len(epsilon_values))
  
    for i in range(0,len(epsilon_values)):
        
        if policy==True:
            epsilon_iterations = mdp.ValueIteration(P, R,0.99,max_iter=max_iterations,epsilon=epsilon_values[i])
        else:
            epsilon_iterations = mdp.PolicyIterationModified(P, R, 0.99,max_iter=max_iterations,epsilon=epsilon_values[i])
            #epsilon_iterations = mdp.PolicyIteration(P, R, 0.99,max_iter=max_iterations,epsilon=epsilon_values[i])
        epsilon_iterations.run()
        #print("Epilson Value: " +str(epsilon_values[i]) +" Iterations to Converge:", discount_iterations.iter)
        #print("Epilson Value: " +str(epsilon_values[i]) +" Time:", discount_iterations.time)
        policies.append(epsilon_iterations.policy)
        iterations.append(epsilon_iterations.iter)
        times.append(epsilon_iterations.time)
        max_value.append(np.amax(epsilon_iterations.V))
    
    return policies,iterations,times,epsilon_values, max_value


def plot_models_epsilon(value_iterations,policy_iterations,value_times,policy_times,epsilon_values, caller_name):
    # if not os.path.exists("Epsilon"):
    #     os.mkdir("Epsilon")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(epsilon_values, value_iterations, label="Value Epsilon")
    ax1.plot(epsilon_values, policy_iterations, label="Policy Epsilon")
    ax1.set_xlabel("Epsilon Value")
    ax1.set_ylabel("Number of Iterations")
    ax1.legend()

    ax2.plot(epsilon_values, value_times, label="Value Epsilon")
    ax2.plot(epsilon_values, policy_times, label="Policy Epsilon")
    ax2.set_xlabel("Epsilon Value")
    ax2.set_ylabel("Time")
    ax2.legend()
    plt.savefig("4."+caller_name+'_Epsilon_Iterations_CPU.png')
    #plt.show()

def plot_models_discount(value_iterations,policy_iterations,value_times,policy_times,discounts, caller_name):
    # if not os.path.exists("Discounts"):
    #     os.mkdir("Discounts")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(discounts, value_iterations, label="Value iterations")
    ax1.plot(discounts, policy_iterations, label="Policy iterations")
    ax1.set_xlabel("Gamma (Discount) Factor")
    ax1.set_ylabel("Number of Iterations")
    ax1.legend()

    ax2.plot(discounts, value_times, label="Value Iterations")
    ax2.plot(discounts, policy_times, label="Policy Iterations")
    ax2.set_xlabel("Gamma (Discount) Factor")
    ax2.set_ylabel("Time")
    ax2.legend()
    plt.savefig("5."+caller_name+'_Gamma_Iterations_CPU.png')
    #plt.show()

def plot_models_maxVal(max_value_vi, max_value_pi, epsilon_values, 
                       max_value_dis_vi, max_value_dis_pi, discounts,
                       caller_name):
    # if not os.path.exists("Discounts"):
    #     os.mkdir("Discounts")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(epsilon_values, max_value_vi, label="Max Value - Value Iterations")
    ax1.plot(epsilon_values, max_value_pi, label="Max Value - Policy Iterations")
    ax1.set_xlabel("Epsilon")
    ax1.set_ylabel("Max Value")
    ax1.legend()

    ax2.plot(discounts, max_value_dis_vi, label="Max Value - Value Iterations")
    ax2.plot(discounts, max_value_dis_pi, label="Max Value - Policy Iterations")
    ax2.set_xlabel("Gamma (Discount) Factor")
    ax2.set_ylabel("Max Value")
    ax2.legend()
    plt.savefig("11."+caller_name+'Epsilon_vs_Max_val.png')
    #plt.show()



def params_iterations(P, R, max_iterations,caller_name):
    value_pol=True
    
    value_policies = []
    value_iterations = []
    value_times = []
    policy_policies = []
    policy_iterations = []
    policy_times = []
   
    value_policies, value_iterations,value_times, discounts, max_value_dis_vi = params_discount_iterations(True, P, R, max_iterations)
    policy_policies, policy_iterations, policy_times,discounts, max_value_dis_pi = params_discount_iterations(False, P, R, max_iterations)
    plot_models_discount(value_iterations,policy_iterations,value_times,policy_times,discounts, caller_name)
    value_policies, value_iterations,value_times, epsilon_values, max_value_eps_vi = params_epsilon_iterations(True, P, R, max_iterations)
    policy_policies, policy_iterations, policy_times,epsilon_values, max_value_eps_pi = params_epsilon_iterations(False, P, R, max_iterations)
    plot_models_epsilon(value_iterations,policy_iterations,value_times,policy_times,epsilon_values, caller_name)
    
    plot_models_maxVal(max_value_eps_vi, max_value_eps_pi, epsilon_values, 
                       max_value_dis_vi, max_value_dis_pi, discounts,
                       caller_name) 
    
    
    vi = mdp.ValueIteration(P, R, 0.10,max_iter=max_iterations)
    pi = mdp.PolicyIterationModified(P, R, 0.10,max_iter=max_iterations)
    
    vi.run()
    print("Value Iterations:")
#     print("Optimal Value Functions:", vi.V)
#     print("optimal policy:", vi.policy)
    print("Iterations to Converge:", vi.iter)
    print("Time:", vi.time)
    
    pi.run()
   
    print("Policy Iterations:")
#   print("Optimal Value Functions:", pi.V)
#   print("optimal policy:", pi.policy)
    print("Iterations to Converge:", pi.iter)
    print("Time:", pi.time)

#     plot_optimal_policy(value_policies,discounts)
    if value_policies== policy_policies: # checks policies are eqaul
        print("Value and Policy are Equal!")
    else:
        print("Value and Policy are NOT Equal!") 
    print("Done!")
