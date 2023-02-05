# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 18:02:57 2023

@author: Florian Martin
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

env = gym.make("FrozenLake-v0", is_slippery=False) # Stochastic Env if True
nS = env.observation_space.n # Number of states
nA = env.action_space.n      # Number of actions

def mapping(policy):
    for state, _ in enumerate(policy):
        if np.argmax(policy[state]) == 0:
            action = 'LEFT'
        elif np.argmax(policy[state]) == 1:
            action = 'DOWN'
        elif np.argmax(policy[state]) == 2:
            action = 'RIGHT'
        elif np.argmax(policy[state]) == 3:
            action = 'UP'
        
        print(f"At state {state}, the optimal action is {action}.")

def policy_evaluation(env, policy, gamma = 1., theta = 1e-8):
    
    V = np.zeros(nS, dtype=np.float64)
    
    while True :
        delta = 0
        for state in range(nS):
                tmp = 0.
                for action, a_prob in enumerate(policy[state]):
                    for tr_prob, next_state, reward, done in env.P[state][action]:
                        tmp += a_prob*tr_prob*(reward + gamma*V[next_state])
                
                delta = max(delta, np.abs(tmp - V[state]))
                V[state] = tmp
                
                
        if delta < theta : 
            break
                
    return V

def policy_improvement(env, V, gamma = 1.):
    
    policy = np.zeros((nS, nA), dtype=np.float64)
    
    for state in range(nS):
        tmp = [0]*nA
        for action in range(nA):
            for tr_prob, next_state, reward, done in env.P[state][action]:
                        tmp[action] += tr_prob*(reward + gamma*V[next_state])
    
        chosen_action = np.argwhere(tmp==np.max(tmp)).flatten()
        policy[state] = np.sum([np.eye(nA)[i] for i in chosen_action], axis=0)/len(chosen_action)
        
    return policy



policy = np.ones((nS, nA))/nA # Uniformly Random Policy

count = 0

GAMMA = 0.1
THETA = 1e-20

while True and count < 10000:
    t0 = time.time()
    V = policy_evaluation(env, policy, gamma=GAMMA, theta=THETA)
    updated_policy = policy_improvement(env, V, gamma=GAMMA)
    t1 = time.time()
    count += 1
    print(f"Iteration {count} performed in {t1-t0}s")
    
    delta = np.abs(policy_evaluation(env, updated_policy, gamma=GAMMA, theta=THETA) - V)
    if np.max(delta) < 1e-6:
        break
    
    policy = np.copy(updated_policy)
    
    
print(f"The policy is now stable after {count} iteration(s)")

plt.figure(figsize=(5, 16))
sns.heatmap(policy,  cmap="YlGnBu", annot=True, cbar=False, square=True)

mapping(policy)


