# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 09:57:13 2022

@author: Florian Martin


Sarsa with Mountain Car Env from Gym
"""





import numpy as np
import gym
import matplotlib.pyplot as plt
from matplotlib import style
import os
import imageio
from PIL import Image
import PIL.ImageDraw as ImageDraw  




env = gym.make("MountainCar-v0")
#env.reset() # Anytime we have an env, reset it


EPISODES = 25_000
SHOW_EVERY = 500

lr = 0.1
gamma = 0.95
epsilon = 0.2
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING   = EPISODES//2
DECAY_RATE = epsilon/(END_EPSILON_DECAYING-START_EPSILON_DECAYING)





"""

The environement is a 2D world.

There are 3 actions in this env :
    - action0 : push car left
    - action1 : do nothing
    - action2 : push car right


observations are (position, velocity)

"""


discrete_obs_space_size = [20] *len(env.observation_space.high) # [20,20] -> 20 separations for each observations
discrete_obs_range_step = (env.observation_space.high-env.observation_space.low)/discrete_obs_space_size

# Initialization of Q-Table
# Rewards are always a negative 1 until you reach the flag, where the reward is 0
# => Initialization in the negative bc for sure the expected reward will be negative

q_table = np.random.uniform(low=-20, high=0, size=(discrete_obs_space_size)+[env.action_space.n]) 

ep_rewards = list()
dic_ep_rewards = {"episode" : [], "average" : [], "min" : [], "max" : []}

# Convert the new_state into discrete

def get_discrete_state(state) :
    discrete_state = (state-env.observation_space.low)/discrete_obs_range_step
    return tuple(discrete_state.astype(np.int32))


for episode in range(EPISODES) :
    
    episode_reward = 0
    
    if episode % SHOW_EVERY == 0 :
        render = True
        
    else :
        render = False
        
    discrete_state = get_discrete_state(env.reset())
    done = False
    
    if np.random.random() < epsilon :
        action = np.random.randint(env.action_space.n, dtype=int) # Epsilon Greedy algorithm
    else :
        action = np.argmax(q_table[discrete_state]) 

    while not done : 

        new_state, reward, done, _ = env.step(action) # observation, reward, terminated
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)
        if render :
            frame = env.render(mode='rgb_array')
            
        # Next Action
        if np.random.random() < epsilon :
            next_action = np.random.randint(env.action_space.n, dtype=int) # Epsilon Greedy algorithm
        else :
            next_action = np.argmax(q_table[new_discrete_state]) 
        
        if not done : 
            q_table[discrete_state + (action, )] = (1-lr)*q_table[discrete_state + (action,)] + lr * (reward + gamma * q_table[new_discrete_state + (next_action,)])
            
        elif new_state[0] >= env.goal_position :
            print(f"Task is achieved on episode {episode} !")
            q_table[discrete_state + (action,)] = 0
            
        discrete_state = new_discrete_state
        action = next_action
        
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING :
        epsilon -= DECAY_RATE
        
    ep_rewards.append(episode_reward)
    
    average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
    dic_ep_rewards["episode"].append(episode)
    dic_ep_rewards["average"].append(average_reward)
    dic_ep_rewards["min"].append(min(ep_rewards[-SHOW_EVERY:]))
    dic_ep_rewards["max"].append(max(ep_rewards[-SHOW_EVERY:]))
    if not episode % SHOW_EVERY :
        print(f"Episode: {episode}, average: {average_reward}, min: {min(ep_rewards[-SHOW_EVERY:])} , max: {max(ep_rewards[-SHOW_EVERY:])}")
    
env.close()

import pickle

with open('reward_sarsa.pickle', 'wb') as file:
    pickle.dump(dic_ep_rewards, file)



