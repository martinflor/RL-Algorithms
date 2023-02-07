# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 10:39:04 2023

@author: Florian Martin

"""

import pickle
import matplotlib.pyplot as plt
import matplotlib

with open('reward_sarsa.pickle', 'rb') as file:
    reward_sarsa = pickle.load(file)
    
with open('reward_q_learning.pickle', 'rb') as file:
    reward_q_learning = pickle.load(file)
    
with open('reward_expected_sarsa.pickle', 'rb') as file:
    reward_expected_sarsa = pickle.load(file)
    
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')
large = 22; med = 16; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")
    
plt.figure(figsize=(12,9))
plt.plot(reward_sarsa["episode"], reward_sarsa["average"], label = "SARSA", color = "purple")
plt.plot(reward_expected_sarsa["episode"], reward_expected_sarsa["average"], label = "Expected SARSA", color = "orange")
plt.plot(reward_q_learning["episode"], reward_q_learning["average"], label = "Q-Learning", color = "red")
plt.grid(alpha=0.25)
plt.legend()
plt.xlabel("Episodes")
plt.ylabel("Average Reward")
plt.title("Comparison between SARSA and Q-Learning on Moutain Car Problem")
plt.savefig("sarsa-q-learning.svg")
plt.show()