# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 09:57:13 2022

@author: Florian Martin


Q-Learning with Mountain Car Env from Gym
"""





import numpy as np
import gym
import matplotlib.pyplot as plt
from matplotlib import style
import os
import imageio
from PIL import Image
import PIL.ImageDraw as ImageDraw  


def _label_with_episode_number(frame, episode_num):
    im = Image.fromarray(frame)

    drawer = ImageDraw.Draw(im)

    if np.mean(im) < 128:
        text_color = (255,255,255)
    else:
        text_color = (0,0,0)
    drawer.text((im.size[0]/20,im.size[1]/18), f'Episode: {episode_num+1}', fill=text_color)

    return im



frames = []

env = gym.make("MountainCar-v0")
#env.reset() # Anytime we have an env, reset it


lr = 0.1
gamma = 0.95
EPISODES = 25_000

SHOW_EVERY = 500
ANIM = False


"""

The environement is a 2D world.

There are 3 actions in this env :
    - action0 : push car left
    - action1 : do nothing
    - action2 : push car right


observations are (position, velocity)

"""


epsilon = 0.2
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING   = EPISODES//2
DECAY_RATE = epsilon/(END_EPSILON_DECAYING-START_EPSILON_DECAYING)


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

    while not done : 
        if np.random.random() < epsilon :
            action = np.random.randint(env.action_space.n, dtype=int) # Epsilon Greedy algorithm
        else :
            action = np.argmax(q_table[discrete_state]) # push car to the right
        new_state, reward, done, _ = env.step(action) # observation, reward, terminated
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)
        if render :
            frame = env.render(mode='rgb_array')
            if ANIM :
                frames.append(_label_with_episode_number(frame, episode_num=episode))
                np.save(f"q_tables_npy/{episode}-qtable.npy", q_table)
        
        if not done : 
            max_future_q = np.max(q_table[new_discrete_state]) #update of the Q-table
            current_q = q_table[discrete_state + (action,)]
            
            new_q = (1-lr)*current_q + lr * (reward + gamma * max_future_q)
            q_table[discrete_state + (action, )] = new_q
            
        elif new_state[0] >= env.goal_position :
            print(f"Task is achieved on episode {episode} !")
            q_table[discrete_state + (action,)] = 0
            
        discrete_state = new_discrete_state
        
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

if ANIM :
    imageio.mimwrite(os.path.join('./video/', 'random_agent.gif'), frames, fps=60)



import pickle

with open('reward_q_learning.pickle', 'wb') as file:
    pickle.dump(dic_ep_rewards, file)







def get_q_color(value, vals):
    if value == max(vals):
        return "green", 1.0
    else:
        return "red", 0.3
    
if ANIM :
    
    fig = plt.figure(figsize=(16, 12))
    
    for i in range(0, EPISODES, SHOW_EVERY):
        print(i)
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
    
        q_table = np.load(f"q_tables_npy/{i}-qtable.npy")
    
        for x, x_vals in enumerate(q_table):
            for y, y_vals in enumerate(x_vals):
                ax1.scatter(x, y, c=get_q_color(y_vals[0], y_vals)[0], marker="o", alpha=get_q_color(y_vals[0], y_vals)[1])
                ax2.scatter(x, y, c=get_q_color(y_vals[1], y_vals)[0], marker="o", alpha=get_q_color(y_vals[1], y_vals)[1])
                ax3.scatter(x, y, c=get_q_color(y_vals[2], y_vals)[0], marker="o", alpha=get_q_color(y_vals[2], y_vals)[1])
    
                ax1.set_ylabel("Velocity")
                ax2.set_ylabel("Velocity")
                ax3.set_ylabel("Velocity")
                
                #ax1.set_xlabel("Position")
                #ax2.set_xlabel("Position")
                ax3.set_xlabel("Position")
                
                ax1.set_title("Action 0")
                ax2.set_title("Action 1")
                ax3.set_title("Action 2")
    
        #plt.show()
        plt.savefig(f"q_tables/{i}.png")
        plt.clf()
        
    q_tables_frame = [] 
    for i in range(0, EPISODES, SHOW_EVERY):
        print(i)
        img = imageio.imread(f"q_tables/{i}.png")
        q_tables_frame.append(Image.fromarray(img))
        
    for i in range(len(q_tables_frame)) :
        img = imageio.imread(f"q_tables/{EPISODES-SHOW_EVERY}.png")
        q_tables_frame.append(Image.fromarray(img))
        
        
    
    imageio.mimwrite(os.path.join('./video/', 'q_tables.gif'), q_tables_frame, fps=10)


def q_table_plot() :

    style.use('ggplot')
    
    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    
    
    for x, x_vals in enumerate(q_table):
        for y, y_vals in enumerate(x_vals):
            ax1.scatter(x, y, c=get_q_color(y_vals[0], y_vals)[0], marker="o", alpha=get_q_color(y_vals[0], y_vals)[1])
            ax2.scatter(x, y, c=get_q_color(y_vals[1], y_vals)[0], marker="o", alpha=get_q_color(y_vals[1], y_vals)[1])
            ax3.scatter(x, y, c=get_q_color(y_vals[2], y_vals)[0], marker="o", alpha=get_q_color(y_vals[2], y_vals)[1])
    
            ax1.set_ylabel("Velocity")
            ax2.set_ylabel("Velocity")
            ax3.set_ylabel("Velocity")
            
            #ax1.set_xlabel("Position")
            #ax2.set_xlabel("Position")
            ax3.set_xlabel("Position")
            
            ax1.set_title("Action 0")
            ax2.set_title("Action 1")
            ax3.set_title("Action 2")
            
    plt.show()

def cluster_plot() :
    
    
    best1_pos = []
    best1_speed = []
    best2_pos = []
    best2_speed = []
    best3_pos = []
    best3_speed = []
    
    for x, x_vals in enumerate(q_table):
        for y, y_vals in enumerate(x_vals):
            idx = np.argmax(y_vals)
            
            if idx == 0 :
                best1_pos.append(x)
                best1_speed.append(y)
                
            if idx == 1 :
                best2_pos.append(x)
                best2_speed.append(y)
                
            if idx == 2 :
                best3_pos.append(x)
                best3_speed.append(y)
            
    #plt.scatter(best1_pos, best1_speed, color = "blue")
    #plt.scatter(best2_pos, best2_speed, color = "orange")
    #plt.scatter(best3_pos, best3_speed, color = "purple")
    
    fig = plt.figure(figsize=(16,16))
    ax = fig.add_subplot(111, projection='3d')
    
    x1, y1, z1 = np.zeros(len(best1_pos)), best1_pos, best1_speed
    x2, y2, z2 = np.zeros(len(best2_pos))+1, best2_pos, best2_speed
    x3, y3, z3 = np.zeros(len(best3_pos))+2, best3_pos, best3_speed
    
    ax.scatter(x1,y1,z1, marker='o')
    ax.scatter(x2,y2,z2, marker='o')
    ax.scatter(x3,y3,z3, marker='o')
    
    ax.set_xlabel("Actions")
    ax.set_ylabel("Position")
    ax.set_zlabel("Speed")
    
    ax.set_xticks([0,1,2])
    
    nb = 360
    for i in range(nb) :
        ax.view_init(elev=10., azim=i)
        plt.savefig(f"q_tables_3D\{i}.png")
    
    tmp_frames = []
    for i in range(nb) :
        img = imageio.imread(f"q_tables_3D/{i}.png")
        tmp_frames.append(Image.fromarray(img))
    
    imageio.mimwrite(os.path.join('./video/', 'q_tables_3D.gif'), tmp_frames, fps=10)
    #plt.show()

def agent_curves() :

    plt.figure(figsize=(12,9))    
    plt.plot(dic_ep_rewards["episode"], dic_ep_rewards["average"], label ="Average")
    plt.plot(dic_ep_rewards["episode"], dic_ep_rewards["min"], label = "Min")
    plt.plot(dic_ep_rewards["episode"], dic_ep_rewards["max"], label ="Max")
    plt.grid(alpha=1)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig("agent_plot/agentCurve.svg")
    plt.show()


    
def _plot() :
    
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(16,16))
    ax = fig.add_subplot(111, projection='3d')
    
    z, x, y = q_table.nonzero()
    
    ax.scatter(x,y,z, c='b', marker='o') 
    
    ax.set_xlabel('Investment')
    ax.set_ylabel('Price')
    ax.set_zlabel('Reward')
    
    plt.show()






    

