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



gamma = 0.5

env = gym.make("MountainCar-v0")

discrete_obs_space_size = [200] *len(env.observation_space.high) # [20,20] -> 20 separations for each observations
discrete_obs_range_step = (env.observation_space.high-env.observation_space.low)/discrete_obs_space_size

def moutainCar(policy):
    
    discrete_state = get_discrete_state(env.reset())
    done = False
    
    while not done : 
    
        action = policy[discrete_state[0], discrete_state[1]]
        new_state, reward, done, _ = env.step(action) # observation, reward, terminated
        new_discrete_state = get_discrete_state(new_state)
        env.render(mode='rgb_array')
            
        if new_state[0] >= env.goal_position :
            print(f"Task is achieved on episode {episode} !")
            break
            
        discrete_state = new_discrete_state 

def get_discrete_state(state) :
    discrete_state = (state-env.observation_space.low)/discrete_obs_range_step
    return tuple(discrete_state.astype(np.int32))

def get_continuous_state(discrete_state) :
    state = env.observation_space.low + discrete_state*discrete_obs_range_step
    return tuple(state.astype(np.float64))

def transition_dynamics(action, xt, vt):
    force = 0.001
    gravity = 0.0025
    vt1 = max(min(vt + (action-1)*force - np.cos(3*xt)*gravity, env.observation_space.high[1]), env.observation_space.low[1])
    xt1 = max(min(xt+vt, env.observation_space.high[0]), env.observation_space.low[0])
    
    return (xt1, vt1)

def transition_probabilities():
    states_to_states_prime = {}
    for i in range(discrete_obs_space_size[0]):
        for j in range(discrete_obs_space_size[1]): # For Loops : (i,j) = state_ij
            for k in range(env.action_space.n):
                xt, vt = get_continuous_state((i,j))
                new_state_ = transition_dynamics(k, xt, vt)
                new_state = get_discrete_state(new_state_)
                reward = -1
                if new_state_[0] >= 0.5:
                    reward = 0
                states_to_states_prime[(i,j, new_state[0], new_state[1], k)] = (reward, 1)
                
    return states_to_states_prime
    


def policy_evaluation(policy, gamma = 0.9, theta = 0.01):
    V = np.zeros((discrete_obs_space_size[0], discrete_obs_space_size[1]), dtype=np.float64)
    delta = 0
    probs = transition_probabilities()
    
    while True :
        for i in range(discrete_obs_space_size[0]):
            for j in range(discrete_obs_space_size[1]):
                v = V[i,j]
                tmp = 0.
                for i_prime in range(discrete_obs_space_size[0]):
                    for j_prime in range(discrete_obs_space_size[1]):
                        if (i, j, i_prime, j_prime, policy[i,j]) in probs:
                            reward, tr = probs[(i, j, i_prime, j_prime, policy[i,j])]
                            tmp +=  tr*(reward+gamma*V[i_prime,j_prime])
                            
                V[i,j] = tmp
                
                delta = max(delta, np.abs(v - V[i,j]))
        
        print(delta)
        if delta < theta : 
            break
                
    return V

def policy_improvement(policy, V):
    
    stable = True
    for i in range(discrete_obs_space_size[0]):
            for j in range(discrete_obs_space_size[1]): # For Loops on state
                old_action = policy[i,j]
                new_v = float('-inf')
                for i_prime in range(discrete_obs_space_size[0]):
                        for j_prime in range(discrete_obs_space_size[1]): # For Loop on state prime
                            for action in range(env.action_space.n):
                                    try :
                                        tr = tr_prob[(i, j, i_prime, j_prime, action)]
                                    except :
                                        tr = 0.
                            policy[i,j] = np.argmax(tmp)
                            
                            if old_action != policy[i,j] :
                                stable = False
    
    return V, policy, stable


def policy_iteration():
        
    # Initialize a policy randomly
    policy = np.zeros((discrete_obs_space_size[0], discrete_obs_space_size[1]), dtype=int)
    
    for i in range(discrete_obs_space_size[0]):
        for j in range(discrete_obs_space_size[1]):
            action = np.random.randint(env.action_space.n, dtype=int)
            policy[i,j] = action
            
    V = policy_evaluation(policy)
    stable = False
    limit = 100
    
    count = 0
    t0 = time.time()
    while not stable and count < limit :
        
        t0_ = time.time()

        V, policy, stable = policy_improvement(policy, V)
        if stable : break
        V = policy_evaluation(policy)
        count += 1
        
        t1_ = time.time()
        print(f"Iteration {count} : {t1_ - t0_}s")
        #moutainCar(policy)
    
    t1 = time.time()
    print("stable : ", stable)
    
    print(f"Time needed : {t1-t0}s")
    
    return V, policy
    




if __name__ == '__main__':
    V, policy = policy_iteration()

