import sys
import random
import numpy as np
import seaborn as sns 
import pandas as pd
from decimal import Decimal
import networkx as nx
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

'''
Three probabilities: q-value (weight1) + observation as second player (weight2) + neighbor_actions (weight3)
'''

store_rewards = []
num_options = 2
reward_vals = [1] * num_options
temperature = 100
actions_taken = []

# np.random.seed(1233)
all_decisions = [] # All the decisions made by all agents
weight_one = 0 # Q Value
weight_two = 0 # Observation as Second Player
weight_three = 1 # All Neighbors
weights = [weight_one, weight_two, weight_three]
episode_num = 0
# Number of times action one was taken per episode
action_one_per_episode = []
# initial value = 00000
# Rewards: +1 -1
# Second agent random.

# REMOVE BEFORE PUSHING TO GITHUB
# SEARCH KEYWORD REMOVE TO FIND ANYTHING YOU NEED TO REMOVE FROM CODE!!
q_val_second_obs_boltz = []
q_val_second_obs_eps = []
########################################################################
'''
Agent CLASS PARAMETERS:
id: Agent's ID
N: Number of Episodes
p_init: Starting value of epsilon
p_end: Final value of epsilon
'''
class Agent:
    def __init__(self, id, N, p_init=0.9, p_end=0.00001):
        
        # Information for the Epsilon Greedy and Update ############
        self.p_init = p_init # starting value of epsilon
        self.p_end = p_end # final value of epsilon
        self.N = N # Number of Episodes being conducted
        #############################################################
        # Count how many times each state, action has been encountered
        self.time_step = np.zeros((num_options + 1, num_options))
        self.neighbors = [] # Only used during small world and scale free networks
        self.node = None # Only used during small world and scale free networks
        self.id = id
        # num_options+1 -> The 0th option is when the agent is going first.
        self.q_vals = np.zeros((num_options+1, num_options))
        # Calculating the q vals over time for all actions  by first agent
        self.q = [[] for _ in range(num_options)]
        
        self.count_action_one = 0
        # Second Player's Observation of First Player's Actions
        self.second_player_obs = []
        # All decisions made by agent
        self.action = []
        # All decisions made by the agent as first player
        self.first_player_action = []


    def select_option(self, state:int, time_step:int, k_last_obsv=0, adj_agents=[], exploration_type="boltzmann"):
        """
            Helps the agent select an option based on the state of the agent.
        
            Parameters:
            state: The state of the agent
            time_step: The current time step
            k_last_obsv: Number of previous episodes to look at as second player
            adj_agents: List of adjacent agents (i.e. neighbors)
            exploration_type: Type of exploration to use (boltzmann or epsilon_greedy)
            
            Returns:
            selection: The option that the agent will select
            q_vals: The q values of each option
        """

        global episode_num, num_diff
        
        # Boltzmann Exploration Purposes #############################
        
        numerator_vals = np.exp(self.q_vals[state] / temperature)
        boltz_prob = []
        
        ##############################################################
        
        # Epsilon Greedy Purposes ####################################
        
        eps_prob = []
        
        ##############################################################
        
        # Q values (FOR TESTING PURPOSES) ############
        q_vals = []
        for index in range(len(self.q_vals[state])):
            q_vals.append(self.q_vals[state][index])
        ##############################################
        
        # P1: Probabilities of each option being selected based on Exploration. ###############################
        
        if exploration_type == "boltzmann":
            denominator_vals = np.sum(numerator_vals)
            boltz_prob = numerator_vals / denominator_vals
            
        if exploration_type == "epsilon_greedy":
            # r = max(1-(time_step/(self.N)), 0) # Original
            # You can change the denominator value to see what value works best.
            t_me = 200 # Time Step to reach the final epsilon value
            r = max(1-(time_step/t_me), 0)

            epsilon = (self.p_init - self.p_end)*r + self.p_end
            # eps_prob = [epsilon/num_options for _ in range(num_options)]
            # Index of the maximum q value
            max_q_index = np.argmax(q_vals)
            if(q_vals[max_q_index] == 0):
                max_q_index = np.random.choice(num_options)
            # if np.random.rand() > epsilon:
            # eps_prob[max_q_index] += 1-epsilon
            q_prob = np.zeros(num_options)
            q_prob[max_q_index] = 1
                
            # Normalize: (In case random < epsilon)
            # eps_prob = np.array(eps_prob)
            # eps_prob = eps_prob/np.sum(eps_prob)
        #######################################################################################################
        
        # The first agent will have a different method of selecting an action from the second agent
        # They must also have made at least two observations of other agents.
        if state == 0 and len(self.second_player_obs)>2:
            # Calculating the q vals over time for agent one
            for i in range(num_options):
                self.q[i].append(q_vals[i])

            # Initialize the selection that the agent will make
            selection = 0
            # Find the probability each option being selected separately
            probs = np.array([])

            for option_num in range(num_options):                
                # Compute three probabilities to determine the overall probability of a certain action being
                # selected by first agent.
                
                # 1. Probabilities obtained through the Exploration Technique
                # (Shown above.)
                    
                # 2. Fraction of time as second player, the agent has seen first player adopt option we are currently observing
                k_last_agent_two_obs = 0
                k_observation = []
                if k_last_obsv!=0 and len(self.second_player_obs) != 0:
                    obs_size = len(self.second_player_obs)
                    if obs_size>k_last_obsv:
                        k_observation = self.second_player_obs[obs_size-k_last_obsv:] # k last observations
                    else:
                        k_observation = self.second_player_obs
                    num_time_chosen = k_observation.count(option_num)
                    k_last_agent_two_obs = float(num_time_chosen/len(k_observation))
                # 3. Percentage of Sampled members who choose action we are currently observing 
                neighbor_percentage = 0.5
                if episode_num>0:
                    # Get the decisions made by all the neighbors last episode
                    neighbor_decisions = []
                    # ASK: IF NEIGHBOR HASN'T MADE A DECISION, WHAT SHOULD MAKE THE ACTION BE COUNTED AS?
                    for agent in adj_agents:
                        if len(agent.action) == 0:
                            # Empty: This Neighbor Hasn't Made a Decision Yet.
                            continue
                        else:
                            neighbor_decisions.append(agent.first_player_action[-1])
                    num_neighbors_chose = neighbor_decisions.count(option_num)
                    neighbor_percentage = float(num_neighbors_chose/len(neighbor_decisions))
                # print("Q-val: ", eps_prob[option_num], "K-Last-Agent-Two-Obs: ", k_last_agent_two_obs, "Sampled-Percentage: ", neighbor_percentage)
                
                # Calculate the Probability of the Option being Chosen
                curr_option_prob = None
                if exploration_type == "boltzmann":
                    curr_option_prob = boltz_prob[option_num]*weights[0] + k_last_agent_two_obs*weights[1] + neighbor_percentage*weights[2]
                elif exploration_type == "epsilon_greedy":
                    curr_option_prob = q_prob[option_num]*weights[0] + k_last_agent_two_obs*weights[1] + neighbor_percentage*weights[2]
                    weighted_prob = curr_option_prob
                    curr_option_prob = (1-epsilon)*weighted_prob + epsilon/num_options
                probs = np.append(probs, curr_option_prob)
            # Normalize the probabilities
            denominator_vals = sum(probs)
            probabilities = probs / denominator_vals
            selection = np.random.choice(num_options, p=probabilities)

        else:
            # REMOVE: FOR TESTING PURPOSES #########################
            global q_val_second_obs_boltz, q_val_second_obs_eps
            if state == 0:
                for i in range(num_options):
                    self.q[i].append(q_vals[i])
            ########################################################
            selection = None
            if exploration_type == "boltzmann":
                # REMOVE: FOR TESTING PURPOSES #########################
                if state !=0:
                    q_val_second_obs_boltz.append(q_vals[1])
                ########################################################
                selection = np.random.choice(num_options, p=boltz_prob)
            elif exploration_type == "epsilon_greedy":
                # REMOVE: FOR TESTING PURPOSES #########################
                if state !=0:
                    q_val_second_obs_eps.append(q_vals[1])
                ########################################################
                second_eps_prob = (1-epsilon)*q_prob + epsilon/num_options
                selection = np.random.choice(num_options, p=second_eps_prob)  
                
        # Graphing Purposes
        # Counts number of times action one is selected as the first player.
        if (state==0) and (selection==1):
            self.count_action_one += 1
        return selection, q_vals

    def update_q(self, state, action, reward, time_step):
        time_step = self.time_step[state][action]
        self.time_step[state][action] += 1
        alpha = 1 / (time_step + 1)
        self.q_vals[state][action] = (1-alpha) * self.q_vals[state][action] + alpha * reward

# BOX PLOT: 
# This will plot out the number of agents whose neighbors all have the same value for action 1 or action 2
# The third bar plot will be for the neither
# Column 1) All Neighbors have Action 1
# Column 2) All Neighbors have Action 2
# Column 3) Neighbors have Different Actions

def plot_neighborhood_decisions(adj_agent_list, agent_list):
    same_actions = []
    for index in range(len(agent_list)):
        agent = agent_list[index]
        adj_agents = adj_agent_list[agent]
        all_same = True
        for adj_agent in adj_agents:
            if adj_agent.action[-1] != agent.action[-1]:
                all_same = False
                break
        if all_same:
            if agent.action[-1] == 0:
                same_actions.append(0)
            else:
                same_actions.append(1)
        else:
            same_actions.append(2)
    # Create Bar Plot
    df = pd.DataFrame(same_actions, columns=['Actions'])
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Actions', data=df, palette='pastel')
    plt.title("Number of Agents Whose Neighbors All Have the Same Value for Action 1 or Action 2")
    plt.xlabel("Actions")
    plt.ylabel("Number of Agents")
    plt.show()

# Pairing in a Random Manner
def pair_agents(agent_list) -> list:
    random.shuffle(agent_list)
    pairs = [(agent_list[i], agent_list[i+1]) for i in range(0, len(agent_list), 2)]
    return pairs

# Create a Matrix of Agents (Randomized)
def create_agents_matrix(agent_list, matrix_row=0, matrix_col=0, type="grid") -> list:
    num_agents = len(agent_list)
    if type=="grid" or type=="all_connect":
        matrix_size = matrix_row*matrix_col
        if matrix_size != num_agents:
            print("The Matrix Dimensions does not Match the number of Agents.")
            sys.exit()
        random.shuffle(agent_list)
        agent_matrix = []
        index = 0
        for _ in range(matrix_row):
            agent_row = []
            for _ in range(matrix_col):
                agent_row.append(agent_list[index])
                index += 1
            agent_matrix.append(agent_row)
    
        return agent_matrix
    if type=="small_world":
        # k: Number of nearest neighbors
        # p: Probability of rewiring each edge
        nodes = nx.watts_strogatz_graph(num_agents, k=4, p=0.1)
        node_list = list(nodes.nodes)

        for i in range(num_agents):
            agent_list[i].node = node_list[i]
            neighbors = list(nodes.neighbors(node_list[i]))
            for agent_id in neighbors:
                agent_list[i].neighbors.append(agent_list[agent_id]) # Add the agent object to the neighbor list
        return agent_list
        
    if type == "scale_free":
        nodes = nx.barabasi_albert_graph(num_agents, m=2) # m: Number of edges to attach from a new node to existing nodes
        node_list = list(nodes.nodes)
        for i in range(num_agents):
            agent_list[i].node = node_list[i]
            neighbors = list(nodes.neighbors(node_list[i]))
            for agent_id in neighbors:
                agent_list[i].neighbors.append(agent_list[agent_id])
        return agent_list

    return []

# num_connect: Number of agents that can be connected to each agent
# If four, then each agent can be paired with one of the four possible agents
# This is above, below, left, and right of the agent
# If eight, then each agent can be paired with one of the eight possible agents
# This includes diagonals as well.
def matrix_pairing(input_matrix, num_connect, one_d_list=None, type="grid"):
    '''
    Returns
    paired_agents: List of paired agents (random)
    adj_agent_list: Dictionary of agents with their list of neighbors
    neighbors: Dictionary of agents with (row, col) of their neighbors
    '''
    paired_agents = []
    agent_neighbors = {} # Dictionary of agents with their list of neighbors
    if type == "grid":
        agent_matrix = input_matrix.copy()
        num_row = len(agent_matrix)
        num_col = len(agent_matrix[0])
        
        for row in range(num_row):
            for col in range(num_col):
                # Randomly select the agents from one of the num_connect options
                options = []
                # num_connect: Number of neighbor
                # Find all the neighboring agents and put into the options list
                if num_connect == 4:
                    options = [((row-1+num_row)%num_row, col), ((row+1)%num_row, col), (row, (col-1+num_col)%num_col), (row, (col+1)%num_col)]
                elif num_connect == 8:
                    options = [((row-1+num_row)%num_row, col), ((row+1)%num_row, col), (row, (col-1+num_col)%num_col), (row, (col+1)%num_col), 
                                ((row-1+num_row)%num_row, (col-1+num_col)%num_col), ((row-1+num_row)%num_row, (col+1)%num_col), 
                                ((row+1)%num_row, (col-1+num_col)%num_col), ((row+1)%num_row, (col+1)%num_col)]
                else:
                    print("Choose either 4 or 8 for the number of connections.")
                    sys.exit()
                agent = agent_matrix[row][col]
                if agent not in agent_neighbors:
                        agent_neighbors[agent] = []
                # Append all the neighbors of the agent
                for option in options:
                    (r, c) = option
                    agent_neighbors[agent].append(agent_matrix[r][c])

                # Randomly select one of the neighbors  
                neighbor = random.choice(agent_neighbors[agent])
                paired_agents.append((agent, neighbor))         
    # All the agents are connected to each other as neighbors
    if type == "all_connect":
        agent_neighbors = {}
        paired_agents = []
        for row in range(len(input_matrix)):
            for col in range(len(input_matrix[0])):
                agent = input_matrix[row][col]
                agent_neighbors[agent] = one_d_list.copy()
                agent_neighbors[agent].remove(agent)
                # Randomly pair agents with their neighbors
                paired_agents.append((agent, random.choice(agent_neighbors[agent])))
    
    if type == "small_world":
        # Create a dictionary of agents with their neighbors
        agent_neighbors = {}
        for agent in input_matrix:
            agent_neighbors[agent] = agent.neighbors
        # Randomly pair agents with their neighbors
        paired_agents = []
        for agent in input_matrix:
            paired_agents.append((agent, random.choice(agent_neighbors[agent])))
        # There are two agent neighbors since adj_agent_list and neighbors are the same for small world
    
    if type == "scale_free":
        # Create a dictionary of agents with their neighbors
        agent_neighbors = {}
        for agent in input_matrix:
            agent_neighbors[agent] = agent.neighbors
        # Randomly pair agents with their neighbors
        paired_agents = []
        for agent in input_matrix:
            paired_agents.append((agent, random.choice(agent_neighbors[agent])))
        # There are two agent neighbors since adj_agent_list and neighbors are the same for scale free
    
    random.shuffle(paired_agents)
    return paired_agents, agent_neighbors  
            

'''
Randomly pairing agents with their neighbors.
This is called for each new episode that is run.
'''
def random_pairing(agent_matrix, neighbors, type="grid") -> list:  
    paired_agents = [] 
    if type=="grid" or type=="all_connect":
        num_row = len(agent_matrix)
        num_col = len(agent_matrix[0])
        for row in range(num_row):
            for col in range(num_col):
                # Randomly select one of the neighbors  
                agent = agent_matrix[row][col]
                neighbor = random.choice(neighbors[agent])
                paired_agents.append((agent, neighbor))    
    if type == "small_world":
        for agent in agent_matrix:
            paired_agents.append((agent, random.choice(neighbors[agent])))
    if type == "scale_free":
        for agent in agent_matrix:
            paired_agents.append((agent, random.choice(neighbors[agent])))   
    return paired_agents

def graph(input_list, title, x_label, y_label, type="plot"):
    if type == "plot":
        plt.plot(input_list)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
    if type == "scatter":
        plt.scatter(range(len(input_list)), input_list)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
        
        
'''
exploration type options: grid (size 4 (Von Neumann Neighborhood)), all_connect
'''

# num_exp_runs: number of experimental runs
def initial_set_run(num_agents:int, num_exp_runs:int, k_last_action_second:int, network_type:str,
            exploration_type="boltzmann", k_last_ep=100, matrix_row=0, matrix_col=0, num_neighbor=0):
    """_summary_

    Args:
        num_agents (int): Number of agents involved in the experiment.
        num_exp_runs (int): Number of experimental runs.
        k_last_action_second (int): Number of previous episodes to look at as second player.
        network_type (str): Type of network to use (e.g. grid, small_world, scale_free).
        exploration_type (str): Type of exploration to utilize. Defaults to "boltzmann". (e.g. boltzmann, epsilon_greedy)
        matrix_row (int): Indicate the dimensions of the matrix row. Defaults to 0.
        matrix_col (int): Indicate the dimensions of the matrix column. Defaults to 0.
        num_neighbor (int):  Defaults to 0.
        k_last_ep (int): K last episodes to check fraction of time each action was taken.

    Returns:
        # TO DO: Before pushing to Github
    """
    global temperature, episode_num, action_one_per_episode, num_options
    # FOR TESTING PURPOSES #######################################################################################
    # Store the q values of each option for each agent (First Agent)
    agent_q_vals = [[[] for _ in range(num_options)] for _ in range(num_agents)]
    
    ##############################################################################################################
    if(num_agents%2 != 0):
        print("The number of agents must be even.")
        sys.exit()
    # One dimensional list of agents
    oned_agent_list = [Agent(i, num_exp_runs) for i in range(num_agents)]
    # two dimensional list of agents
    agent_list = create_agents_matrix(oned_agent_list, matrix_row, matrix_col, type=network_type)
    agent_actions = [[] for _ in range(num_agents)]

    # Check if K last episodes is greater than the number of experimental runs
    if k_last_ep>num_exp_runs:
        k_last_ep = num_exp_runs
    # Keep track of the fraction of times each action was taken over the last k (100) episodes
    frac_opt_chosen_k_last = np.zeros((num_options, k_last_ep), dtype=float)
    option_chosen = [[] for _ in range(num_options)] # Number of times each option was chosen per episode
    
    agent_reward = []
    not_same = 0
    # Store the q values of each option for First Agent.
    q_every_episode = [[] for _ in range(num_options)]

    # Number of times action one was taken by the first agent
    action_one = []
    
    percentage = []
    avg_reward = []
    
    paired_agents, neighbors = matrix_pairing(agent_list, num_neighbor, oned_agent_list, type=network_type)
    # agent_matrix, neighbors, num_row, num_col
    for x in range(num_exp_runs):
        options = np.zeros(num_options) # Number of times each option was chosen in the episode
        # num_connect: Size of neighborhood
        paired_agents = random_pairing(agent_list, neighbors, type=network_type)
        
        # Options taken for the episode by each agent
        agent_options = []
        # Number of times action one was chosen in the episode
        action_one_cnt = 0

        curr_reward = 0
        # Storing the q vals for all options in the Episode
        # It will be stored into q_every_episode
        curr_q_vals = [[] for _ in range(num_options)]

        # For each pair of agents, select an option
        for pair in paired_agents:
            (agent_one, agent_two) = pair
            # Select an option for the first agent
            action_one, q_vals_one = agent_one.select_option(
                                state=0,
                                time_step=x,
                                k_last_obsv=k_last_action_second,
                                adj_agents=neighbors[agent_one],
                                exploration_type=exploration_type
            
                               )
            # For the first episode, the first agent will select the option that belongs to them
            if x==0:
                action_one = agent_one.id
            agent_one.action.append(action_one)
            agent_one.first_player_action.append(action_one)
            # Store the q values for each option for the first agent
            for index in range(len(q_vals_one)):
                curr_q_vals[index].append(q_vals_one[index])

            options[action_one] += 1
            agent_two.second_player_obs.append(action_one)
            action_two, _ = agent_two.select_option(action_one + 1, 
                                                    time_step = x,
                                                    exploration_type=exploration_type)
            
            if action_one == 0:
                action_one_cnt += 1
    
            agent_two.action.append(action_two)
            options[action_two] += 1
            
            # If agent one and two have the same action, then reward is 1.
            # Otherwise, reward is 0.
            if action_one == action_two:
                reward = 1
            else:
                reward = 0
                not_same += 1
            curr_reward += reward
            agent_one.update_q(0, action_one, reward, x) # x: time step
            agent_reward.append(reward)
            agent_two.update_q(action_one + 1, action_two, reward, x) # x: time step
            agent_actions[agent_one.id].append(action_one)
            agent_actions[agent_two.id].append(action_two)
            # Store the actions taken by each agent for neighbor_decision list.
            # For the third probability calculation
            agent_options.append(action_one)
            agent_options.append(action_two)
            
            # FOR TESTING PURPOOSES ########################################################
            # Store the q values for each option for each agent (First Agent)
            for i in range(num_options):
                agent_q_vals[agent_one.id][i].append(q_vals_one[i])
            ################################################################################
        # Keeps track of all agent actions for each episode 
        for option in agent_options:
            all_decisions.append(option)
        
        # Store the q values for each episode for the first agent
        for index in range(len(curr_q_vals)):
            q_every_episode.append(curr_q_vals[index])
        
        if(temperature>0.01):
            temperature *= 0.97

        # Store the number of times each option was chosen in the episode
        for x in range(len(options)):
            option_chosen[x].append(options[x])
        # Store the percentage of action one taken in the episode
        percentage.append(float(action_one_cnt/len(paired_agents))*100)
        # Store the average reward in the episode
        avg_reward.append(curr_reward/(len(paired_agents)))
        episode_num += 1
        # Store the number of times action one was taken in the episode
        action_one_per_episode.append(action_one_cnt)
    # Each agent's q values for each episode
    q_val_agent = []
    for agent in oned_agent_list:
        q_val_agent.append(agent.q) # Q values for each agent
    
    # Fraction of time each action was taken over the last k episodes
    for i in range(num_options):
        frac_opt_chosen_k_last[i] = [x/(num_agents*2) for x in option_chosen[i][-k_last_ep:]]

    # Agent Actions: Dictionary of agents and the actions they have made
    # Option Chosen: Given index of action number, it will show the list of times that action was chosen per episode
    return agent_actions, option_chosen, agent_reward, percentage, avg_reward, q_every_episode, q_val_agent, agent_q_vals, frac_opt_chosen_k_last
                
'''
exploration type options: grid (size 4 (Von Neumann Neighborhood)), all_connect
'''

# num_exp_runs: number of experimental runs
def run_exp(num_agents:int, num_exp_runs:int, k_last_action_second:int, network_type:str,
            exploration_type="boltzmann", k_last_ep=100, matrix_row=0, matrix_col=0, num_neighbor=0):
    """_summary_

    Args:
        num_agents (int): Number of agents involved in the experiment.
        num_exp_runs (int): Number of experimental runs.
        k_last_action_second (int): Number of previous episodes to look at as second player.
        network_type (str): Type of network to use (e.g. grid, small_world, scale_free).
        exploration_type (str): Type of exploration to utilize. Defaults to "boltzmann". (e.g. boltzmann, epsilon_greedy)
        matrix_row (int): Indicate the dimensions of the matrix row. Defaults to 0.
        matrix_col (int): Indicate the dimensions of the matrix column. Defaults to 0.
        num_neighbor (int):  Defaults to 0.
        k_last_ep (int): K last episodes to check fraction of time each action was taken.

    Returns:
        # TO DO: Before pushing to Github
    """
    global temperature, episode_num, action_one_per_episode, num_options
    # FOR TESTING PURPOSES #######################################################################################
    # Store the q values of each option for each agent (First Agent)
    agent_q_vals = [[[] for _ in range(num_options)] for _ in range(num_agents)]
    
    ##############################################################################################################
    if(num_agents%2 != 0):
        print("The number of agents must be even.")
        sys.exit()
    # One dimensional list of agents
    oned_agent_list = [Agent(i, num_exp_runs) for i in range(num_agents)]
    # two dimensional list of agents
    agent_list = create_agents_matrix(oned_agent_list, matrix_row, matrix_col, type=network_type)
    agent_actions = [[] for _ in range(num_agents)]

    # Check if K last episodes is greater than the number of experimental runs
    if k_last_ep>num_exp_runs:
        k_last_ep = num_exp_runs
    # Keep track of the fraction of times each action was taken over the last k (100) episodes
    frac_opt_chosen_k_last = np.zeros((num_options, k_last_ep), dtype=float)
    option_chosen = [[] for _ in range(num_options)] # Number of times each option was chosen per episode
    
    # Detects the First Convergence
    # index 0: If Convergence has been detected: true
    # index 1: Action the agents converged to
    # index 2: The episode number where the convergence was detected
    first_conv = [False, -1, -1]
    
    agent_reward = []
    not_same = 0
    # Store the q values of each option for First Agent.
    q_every_episode = [[] for _ in range(num_options)]

    # Number of times action one was taken by the first agent
    action_one = []
    
    percentage = []
    avg_reward = []
    
    paired_agents, neighbors = matrix_pairing(agent_list, num_neighbor, oned_agent_list, type=network_type)
    # agent_matrix, neighbors, num_row, num_col
    for x in range(num_exp_runs):
        options = np.zeros(num_options) # Number of times each option was chosen in the episode
        # num_connect: Size of neighborhood
        paired_agents = random_pairing(agent_list, neighbors, type=network_type)
        
        # Options taken for the episode by each agent
        agent_options = []
        # Number of times action one was chosen in the episode
        action_one_cnt = 0

        curr_reward = 0
        # Storing the q vals for all options in the Episode
        # It will be stored into q_every_episode
        curr_q_vals = [[] for _ in range(num_options)]

        # For each pair of agents, select an option
        for pair in paired_agents:
            (agent_one, agent_two) = pair
            # Select an option for the first agent
            action_one, q_vals_one = agent_one.select_option(
                                state=0,
                                time_step=x,
                                k_last_obsv=k_last_action_second,
                                adj_agents=neighbors[agent_one],
                                exploration_type=exploration_type
                                )
            agent_one.action.append(action_one)
            agent_one.first_player_action.append(action_one)
            # Store the q values for each option for the first agent
            for index in range(len(q_vals_one)):
                curr_q_vals[index].append(q_vals_one[index])

            options[action_one] += 1
            agent_two.second_player_obs.append(action_one)
            action_two, _ = agent_two.select_option(action_one + 1, 
                                                    time_step = x,
                                                    exploration_type=exploration_type)
            
            if action_one == 0:
                action_one_cnt += 1
    
            agent_two.action.append(action_two)
            options[action_two] += 1
            
            # If agent one and two have the same action, then reward is 1.
            # Otherwise, reward is 0.
            if action_one == action_two:
                reward = 1
            else:
                reward = 0
                not_same += 1
            curr_reward += reward
            agent_one.update_q(0, action_one, reward, x) # x: time step
            agent_reward.append(reward)
            agent_two.update_q(action_one + 1, action_two, reward, x) # x: time step
            agent_actions[agent_one.id].append(action_one)
            agent_actions[agent_two.id].append(action_two)
            # Store the actions taken by each agent for neighbor_decision list.
            # For the third probability calculation
            agent_options.append(action_one)
            agent_options.append(action_two)
            
            # FOR TESTING PURPOOSES ########################################################
            # Store the q values for each option for each agent (First Agent)
            for i in range(num_options):
                agent_q_vals[agent_one.id][i].append(q_vals_one[i])
            ################################################################################
        # Keeps track of all agent actions for each episode 
        for option in agent_options:
            all_decisions.append(option)
        
        # Store the q values for each episode for the first agent
        for index in range(len(curr_q_vals)):
            q_every_episode.append(curr_q_vals[index])
        
        if(temperature>0.01):
            temperature *= 0.97

        # Store the number of times each option was chosen in the episode
        for x in range(len(options)):
            option_chosen[x].append(options[x])
        # Detect First Convergence
        if first_conv[0] == False:
            for index in range(len(options)):
                if options[index]/(num_agents*2) >= 0.9:
                    first_conv[0] = True
                    first_conv[1] = index + 1
                    first_conv[2] = x
                    break
        # Store the percentage of action one taken in the episode
        percentage.append(float(action_one_cnt/len(paired_agents))*100)
        # Store the average reward in the episode
        avg_reward.append(curr_reward/(len(paired_agents)))
        episode_num += 1
        # Store the number of times action one was taken in the episode
        action_one_per_episode.append(action_one_cnt)
    # Each agent's q values for each episode
    q_val_agent = []
    for agent in oned_agent_list:
        q_val_agent.append(agent.q) # Q values for each agent
    
    # Fraction of time each action was taken over the last k episodes
    for i in range(num_options):
        frac_opt_chosen_k_last[i] = [x/(num_agents*2) for x in option_chosen[i][-k_last_ep:]]

    # Agent Actions: Dictionary of agents and the actions they have made
    # Option Chosen: Given index of action number, it will show the list of times that action was chosen per episode
    return agent_actions, option_chosen, agent_reward, percentage, avg_reward, q_every_episode, q_val_agent, agent_q_vals, frac_opt_chosen_k_last, first_conv

####################################################################################

# Find the closest square factors to determine the matrix dimension given number of agents
def closest_square_factors(num_agents):
    smallest_diff = num_agents - 1
    final_pair = (1, num_agents)
    for factor in range(1, int(num_agents**0.5) + 1):
        if num_agents % factor == 0:
            quotient = num_agents // factor
            diff = abs(quotient - factor)
            if diff < smallest_diff:
                smallest_diff = diff
                final_pair = (factor, quotient)
    return final_pair

# Find where the convergence takes place
def find_convergence_point(action_list, k_last_episodes, prob_threshold):
    index = 0
    while (index + k_last_episodes) < len(action_list):
        fraction_list = action_list[index:index+k_last_episodes]
        converges = does_converge(fraction_list, prob_threshold)
        if converges:
            return index
        index += 1
    return -1
        

def clear_all():
    global action_one_per_episode, store_rewards, reward_vals, num_options, temperature, actions_taken, all_decisions, weights, episode_num
    action_one_per_episode = []
    store_rewards = []
    temperature = 100
    actions_taken = []

    # np.random.seed(1233)
    all_decisions = []

    weights = [weight_one, weight_two, weight_three]
    episode_num = 0
    # Number of times action one was taken per episode
    action_one_per_episode = []

def does_converge(frac_k_last, prob_threshold):
    last_episode = [row[-1] for row in frac_k_last]
    pass_prob_thresh = False
    converge_to_one = False
    # Check if any of the fraction of time each action was taken is greater than the probability threshold
    for index in range(len(last_episode)):
        prob = last_episode[index]
        if prob > prob_threshold:
            pass_prob_thresh = True
            if index == 0:
                converge_to_one = True
            break
    return (pass_prob_thresh, converge_to_one)

def reset_option_num():
    global num_options
    num_options = 2


# Minority following different conventions
def minority_adapt():
    global weights
    weights = [1, 0, 0] # Start with 1,0,0 to see what one can get with only action selection mechanism
    
    
def main():    
    # Find the amount of times the graph is converging to action one and two ##############
    # global all_decisions
    num_agents = 100
    num_episodes = 2000
    k_last_ep = num_episodes # Last k episodes to check fraction of time each action was taken

    # q_vals: Q values for each agent
    agent_actions, chosen_option, agent_reward, percentage, avg_reward, q_ops, q_val_agent, q_vals, frac_k_last, first_conv = run_exp(num_agents,
        num_episodes, k_last_action_second=10,k_last_ep=k_last_ep, network_type="all_connect", exploration_type="epsilon_greedy", matrix_row=10,
        matrix_col=10, num_neighbor=8)
    # For Testing Purposes ###################################################################################
    
    # Fraction of time each action was taken over the last k episodes
    x_axis = list(range(num_episodes-k_last_ep+1, num_episodes+1))
    colors = cm.cool(np.linspace(0, 1, num_options))
    colors = ['teal', 'coral']
    fig, ax = plt.subplots()
    for i in range(num_options):
        ax.scatter(x_axis, frac_k_last[i], c=[colors[i]], alpha=1)

    # plt.title("Fraction of Time Each Action was Taken Over the Last " + str(k_last_ep) + " Episodes")
    ax.set_xlabel("Episode Number", fontsize=16)
    ax.set_ylabel("Fraction of Time Taken", fontsize=16)
    
    ax.set_xticks(np.arange(0, num_episodes+1, 500))
    plt.show()
    ########################################################################################################
        
        
    

    


if __name__ == "__main__":
    main()