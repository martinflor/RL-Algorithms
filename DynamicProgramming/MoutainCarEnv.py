# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 10:04:22 2023

@author: Florian Martin 

Reinforcement Learning : Dynamic Programming on MoutainCar Environment

"""

import gym
import numpy as np
import matplotlib.pyplot as plt
import time



gamma = 0.9
REWARD = 100
quantization = 50

env = gym.make("MountainCar-v0")

discrete_obs_space_size = [quantization] *len(env.observation_space.high) # [20,20] -> 20 separations for each observations
discrete_obs_range_step = (env.observation_space.high-env.observation_space.low)/discrete_obs_space_size

nA = env.action_space.n
nS = quantization*quantization

def mountainCar(policy):
    
    discrete_state = get_discrete_state(env.reset())
    done = False
    
    while not done : 
        """
        tmp = policy[discrete_state]
        
        chosen_action = np.argwhere(tmp==np.max(tmp)).flatten()
        if len(chosen_action) > 1:
            action = np.random.choice(chosen_action)
        else:
            action = chosen_action[0]
        """
        action = np.argmax(policy[discrete_state])
        print(action)
        new_state, reward, done, _ = env.step(action) # observation, reward, terminated
        new_discrete_state = get_discrete_state(new_state)
        env.render(mode='rgb_array')
            
        if new_state[0] >= env.goal_position :
            print(f"Task is achieved on episode {episode} !")
            break
            
        discrete_state = new_discrete_state 

def get_discrete_state(state) :
    discrete_state = (state-env.observation_space.low)/discrete_obs_range_step
    discrete_state = discrete_state.astype(np.int32)
    return discrete_state[0]*quantization+discrete_state[1]

def get_continuous_state(discrete_state) :
    discrete_state = (discrete_state//quantization, discrete_state%quantization)
    state = env.observation_space.low + discrete_state*discrete_obs_range_step
    return tuple(state.astype(np.float64))

def transition_dynamics(action, xt, vt):
    force = 0.001
    gravity = 0.0025
    vt1 = max(min(vt + (action-1)*force - np.cos(3*xt)*gravity, env.observation_space.high[1]), env.observation_space.low[1])
    xt1 = max(min(xt+vt, env.observation_space.high[0]), env.observation_space.low[0])
    
    return (xt1, vt1)

def transition_probabilities():
    P = {}
    
    for state in range(nS):
            actions = {}
            for k in range(env.action_space.n):
                xt, vt = get_continuous_state(state)
                next_state_ = transition_dynamics(k, xt, vt) # continuous
                next_state = get_discrete_state(next_state_) # discrete
                
                reward = -1
                if next_state_[0] >= 0.5:
                    reward = 0
                    
                actions[k] = [(1, next_state, reward, True if reward == 0 else False)]
            
            P[state] = actions
                
    return P
    

def policy_evaluation(policy, P, gamma = .9, theta = 1e-4):
    
    V = np.zeros(nS, dtype=np.float64)
    
    while True:
        delta = 0
        for state in range(nS):
                tmp = 0.
                for action, a_prob in enumerate(policy[state]):
                    for tr_prob, next_state, reward, done in P[state][action]:
                        tmp += a_prob*tr_prob*(reward + gamma*V[next_state])
                
                delta = max(delta, np.abs(tmp - V[state]))
                V[state] = tmp
                print(delta)

        if delta < theta : 
            break
                
    return V


def policy_improvement(V, P, gamma = .9):
    
    policy = np.zeros((nS, nA), dtype=np.float64)
    
    for state in range(nS):
        tmp = [0]*nA
        for action in range(nA):
            for tr_prob, next_state, reward, done in P[state][action]:
                        tmp[action] += tr_prob*(reward + gamma*V[next_state])
    
        chosen_action = np.argwhere(tmp==np.max(tmp)).flatten()
        policy[state] = np.sum([np.eye(nA)[i] for i in chosen_action], axis=0)/len(chosen_action)
        
    return policy



policy = np.ones((nS, nA))/nA # Uniformly Random Policy
P = transition_probabilities()
count = 0

while True and count < 10000:
    t0 = time.time()
    V = policy_evaluation(policy, P, gamma = gamma)
    updated_policy = policy_improvement(V, P, gamma = gamma)
    t1 = time.time()
    count += 1
    print(f"Iteration {count} performed in {t1-t0}s")
    
    delta = np.abs(policy_evaluation(updated_policy, P) - V)
    if np.max(delta) < 1e-6:
        break
    
    policy = np.copy(updated_policy)


plt.figure(figsize=(4,16))
plt.imshow(policy)