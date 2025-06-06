import sys
import random
import numpy as np
import seaborn as sns 
import pandas as pd
'''
Three probabilities: q-value (weight1) + observation as second player (weight2) + neighbor_actions (weight3)
'''

epsilon = 0.3
store_rewards = []
reward_vals = [1,1]
num_options = len(reward_vals)
temperature = 100
actions_taken = []

# np.random.seed(1233)
neighbor_decisions = [] # All the decisions made by all agents
weight_one = 0.33
weight_two = 0.33
weight_three = 0.33
weights = [weight_one, weight_two, weight_three]
episode_num = 0
selection_one = []
selection_two = []
# Number of times action one was taken per episode
action_one_per_episode = []
     
class Agent:
    def __init__(self, id):
        self.id = id
        # num_options+1 -> The 0th option is when the agent is going first.
        self.q_vals = np.zeros((num_options+1, num_options))
        # Count how many times each state, action has been encountered
        self.time_step = np.zeros((num_options + 1, num_options))
        # Calculating the q vals over time 
        self.q = []
        self.count_action_one = 0
        # Second Player's Observation of First Player's Actions
        self.second_player_obs = []
        # All decisions made by agent
        self.action = []

    # Utilized Boltzmann Exploration Technique
    # k_episodes: Number of previous episodes to look at
    def select_option(self, state, k_episodes=0, adj_agents=[]):
        global selection_one, selection_two, episode_num, num_diff
        numerator_vals = np.exp(self.q_vals[state] / temperature)
        # Q value of Option One (FOR TESTING PURPOSES)
        q_val_one = self.q_vals[state][0]
        q_val_two = self.q_vals[state][1]
        
        # The first agent will have a different method of selecting an action from the second agent
        if state == 0 and len(self.second_player_obs)>2:
            # Calculating the q vals over time for agent one
            self.q.append(q_val_one)
            # Initialize the selection that the agent will make
            selection = 0
            # Find the probability each option being selected separately
            probs = np.array([])

            for option_num in range(num_options):
                # Compute three probabilities of selection option one by first agent
                # 1. Q value of Option one
                q_val = self.q_vals[state][option_num]
                # 2. Fraction of time as second player, the agent has seen first player adopt option we are currently observing
                k_last_agent_two_obs = 0
                if k_episodes!=0 and len(self.second_player_obs) != 0:
                    obs_size = len(self.second_player_obs)
                    if obs_size>k_episodes:
                        k_observation = self.second_player_obs[obs_size-k_episodes:]
                    else:
                        k_observation = self.second_player_obs
                    num_time_chosen = k_observation.count(option_num)
                    k_last_agent_two_obs = float(num_time_chosen/len(k_observation))
                # 3. Percentage of Sampled members who choose action we are currently observing 
                neighbor_percentage = 0.5
                if episode_num>0:
                    # Get the decisions made by all the neighbors last episode
                    neighbor_decisions = []
                    for agent in adj_agents:
                        if len(agent.action) == 0:
                            print("Empty")
                        neighbor_decisions.append(agent.action[-1])          
                    num_neighbors_chose = neighbor_decisions.count(option_num)
                    neighbor_percentage = float(num_neighbors_chose/len(neighbor_decisions))
                # print("Q-val: ", q_val, "K-Last-Agent-Two-Obs: ", k_last_agent_two_obs, "Sampled-Percentage: ", neighbor_percentage)
                # Calculate the final probability of selecting option one 
                denominator_vals = np.sum(numerator_vals)
                probabilities = numerator_vals / denominator_vals
                final_prob = probabilities[option_num]*weights[0] + k_last_agent_two_obs*weights[1] + neighbor_percentage*weights[2]
                probs = np.append(probs, final_prob)
            # Normalize the probabilities
            # exp_vals = np.exp(probs/temperature)
            denominator_vals = sum(probs)
            probabilities = probs / denominator_vals
            selection = np.random.choice(num_options, p=probabilities)

            # For graphing purposes (Cumulative)
            if selection==0:
                selection_one.append(1)
                selection_two.append(0)
            else:
                selection_two.append(1)
                selection_one.append(0)
        else:
            denominator_vals = np.sum(numerator_vals)
            probabilities = numerator_vals / denominator_vals
            selection = np.random.choice(num_options, p=probabilities)  
        if (state==0) and (selection==1):
            self.count_action_one += 1
        return selection, q_val_one, q_val_two

    def update_q(self, state, action, reward):
        time_step = self.time_step[state][action]
        self.time_step[state][action] += 1
        alpha = 1 / (time_step + 1)
        self.q_vals[state][action] = (1-alpha) * self.q_vals[state][action] + alpha * reward