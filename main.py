import sys
sys.path.append("./simple-car-env-template")


import gym
import simple_driving
# import pybullet_envs
import pybullet as p
import numpy as np
import math
from collections import defaultdict
import pickle
import torch
import random


import gym
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import defaultdict
import pickle
import torch
import random
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

TRAIN = False  # if set to false will skip training, load the last saved model and use that for testing
USE_PREVIOUS_MODEL = False # if set to false will not use the previous model but will use the current model

# Hyper parameters that will be used in the DQN algorithm
EPISODES = 5000                 # number of episodes to run the training for
LEARNING_RATE = 0.00015         # the learning rate for optimising the neural network weights
MEM_SIZE = 50000                # maximum size of the replay memory - will start overwritting values once this is exceed
REPLAY_START_SIZE = 10000       # The amount of samples to fill the replay memory with before we start learning
BATCH_SIZE = 96                 # Number of random samples from the replay memory we use for training each iteration
GAMMA = 0.99                    # Discount factor
EPS_START = 0.1                 # Initial epsilon value for epsilon greedy action sampling
EPS_END = 0.0001                # Final epsilon value
EPS_DECAY = 4 * MEM_SIZE        # Amount of samples we decay epsilon over
MEM_RETAIN = 0.1                # Percentage of initial samples in replay memory to keep - for catastrophic forgetting
NETWORK_UPDATE_ITERS = 5000     # Number of samples 'C' for slowly updating the target network \hat{Q}'s weights with the policy network Q's weights

FC1_DIMS = 192                   # Number of neurons in our MLP's first hidden layer
FC2_DIMS = 128                   # Number of neurons in our MLP's second hidden layer
FC3_DIMS = 64                   # Number of neurons in our MLP's second hidden layer

# metrics for displaying training status
best_reward = 0
average_reward = 0
episode_history = []
episode_reward_history = []
# List to store current_goal for each episode
current_goal_history = []

np.bool = np.bool_
# for creating the policy and target networks - same architecture
class Network(torch.nn.Module):
    def __init__(self, env):
        super().__init__()
        self.input_shape = env.observation_space.shape
        self.action_space = env.action_space.n
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        print("self.device => ", self.device)
        # self.device = torch.device('cuda:0')
        self.to(self.device)
        
        # build an MLP with 2 hidden layers
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(*self.input_shape, FC1_DIMS),   # input layer
            torch.nn.ReLU(),     # activation function
            torch.nn.Linear(FC1_DIMS, FC2_DIMS),    # hidden layer 1
            torch.nn.ReLU(),     # activation function
            torch.nn.Linear(FC2_DIMS, FC3_DIMS),    # hidden layer 2
            torch.nn.ReLU(),     # activation function
            torch.nn.Linear(FC3_DIMS, self.action_space)    # output layer
        )

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()  # loss function

        print(f'Device used: {self.device}')
        print(f'number of action space: {self.action_space}')

    def forward(self, x):
        return self.layers(x)

# handles the storing and retrival of sampled experiences
class ReplayBuffer:
    def __init__(self, env):
        self.mem_count = 0
        print(f'number of observation_space.shape: {env.observation_space.shape[0]}')
        self.states = np.zeros((MEM_SIZE, env.observation_space.shape[0]),dtype=np.float32)
        self.actions = np.zeros(MEM_SIZE, dtype=np.int64)
        self.rewards = np.zeros(MEM_SIZE, dtype=np.float32)
        self.states_ = np.zeros((MEM_SIZE, env.observation_space.shape[0]),dtype=np.float32)
        self.dones = np.zeros(MEM_SIZE, dtype=np.bool)

    def add(self, state, action, reward, state_, done):
        # if memory count is higher than the max memory size then overwrite previous values
        if self.mem_count < MEM_SIZE:
            mem_index = self.mem_count
        else:
            ############ avoid catastropic forgetting - retain initial 10% of the replay buffer ##############
            mem_index = int(self.mem_count % ((1-MEM_RETAIN) * MEM_SIZE) + (MEM_RETAIN * MEM_SIZE))  # avoid catastrophic forgetting, retain first 10% of replay buffer
            ##################################################################################################

        self.states[mem_index]  = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_
        self.dones[mem_index] =  1 - done

        self.mem_count += 1

    # returns random samples from the replay buffer, number is equal to BATCH_SIZE
    def sample(self):
        MEM_MAX = min(self.mem_count, MEM_SIZE)
        batch_indices = np.random.choice(MEM_MAX, BATCH_SIZE, replace=True)

        states  = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones   = self.dones[batch_indices]

        return states, actions, rewards, states_, dones

class DQN_Solver:
    def __init__(self, env):
        self.memory = ReplayBuffer(env)
        self.policy_network = Network(env)  # Q
        self.target_network = Network(env)  # \hat{Q}
        self.target_network.load_state_dict(self.policy_network.state_dict())  # initially set weights of Q to \hat{Q}
        self.learn_count = 0    # keep track of the number of iterations we have learnt for
        self.env = env

    # epsilon greedy
    def choose_action(self, observation):
        # only start decaying epsilon once we actually start learning, i.e. once the replay memory has REPLAY_START_SIZE
        if self.memory.mem_count > REPLAY_START_SIZE:
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * self.learn_count / EPS_DECAY)
        else:
            eps_threshold = 1.0
        # if we rolled a value lower than epsilon sample a random action
        if random.random() < eps_threshold:
            return np.random.choice(np.array(range(9)), p=[0.05,0.05,0.1,0.1,0.1,0.1,0.15,0.2,0.15])
        
        # otherwise policy network, Q, chooses action with highest estimated Q-value so far
        state = torch.tensor(observation).float().detach()
        state = state.unsqueeze(0)
        self.policy_network.eval()  # only need forward pass
        with torch.no_grad():       # so we don't compute gradients - save memory and computation
            ################ retrieve q-values from policy network, Q ################################
            q_values = self.policy_network(state) #by using state
            ##########################################################################################
        return torch.argmax(q_values).item()

    # main training loop
    def learn(self):
        states, actions, rewards, states_, dones = self.memory.sample()  # retrieve random batch of samples from replay memory
        states = torch.tensor(states , dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        states_ = torch.tensor(states_, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)
        batch_indices = np.arange(BATCH_SIZE, dtype=np.int64)

        self.policy_network.train(True)
        q_values = self.policy_network(states)                # get current q-value estimates (all actions) from policy network, Q
        q_values = q_values[batch_indices, actions]           # q values for sampled actions only

        self.target_network.eval()                            # only need forward pass
        with torch.no_grad():                                 # so we don't compute gradients - save memory and computation
            ###### get q-values of states_ from target network, \hat{q}, for computation of the target q-values ###############
            q_values_next = self.target_network(states_)
            ###################################################################################################################
        q_values_next_max = torch.max(q_values_next, dim=1)[0]  # max q values for next state
        q_target = rewards + GAMMA * q_values_next_max * dones  # our target q-value
        ###### compute loss between target (from target network, \hat{Q}) and estimated q-values (from policy network, Q) #########
        loss = self.policy_network.loss(q_target, q_values)
        ###########################################################################################################################

        #compute gradients and update policy network Q weights
        self.policy_network.optimizer.zero_grad()
        loss.backward()
        self.policy_network.optimizer.step()
        self.learn_count += 1

        # set target network \hat{Q}'s weights to policy network Q's weights every C steps
        if  self.learn_count % NETWORK_UPDATE_ITERS == NETWORK_UPDATE_ITERS - 1:
            print("updating target network")
            self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def returning_epsilon(self):
        return self.exploration_rate

def train_model():
    # Start time for Training
    start_time = time.time()
    # create training model for simple driving 
    env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True, render_mode='tp_camera')
    # set manual seeds so we get same behaviour everytime - so that when you change your hyper parameters you can attribute the effect to those changes
    SEED = 0
    env.action_space.seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # setup eqisode
    episode_batch_score = 0
    episode_reward = 0
    agent = DQN_Solver(env)  # create DQN agent
    # plt.clf()
    
    if os.path.exists(previous_model_file):
        # Load pre-trained model
        try:
            if USE_PREVIOUS_MODEL:
                agent.policy_network.load_state_dict(torch.load(previous_model_file))
            else:
                agent.policy_network.load_state_dict(torch.load(model_file))
        except Exception as e:
            print("Error loading pre-trained model:", e)

        print("Pre-trained model loaded successfully!")
    else:
        print("No pre-trained model found. Starting training from scratch...")
        
    for i in range(EPISODES):
        state = env.reset()[0]
        episode_current_goal = 0  # Current goal reached in this episode
        while True:
            # sampling loop - sample random actions and add them to the replay buffer
            action = agent.choose_action(state)
            state_, reward, done, _, info = env.step(action)
            
            # Access current_goal to the episode_current_goal
            episode_current_goal = state_[len(state_)-1]
            
            ####### add sampled experience to replay buffer ##########
            agent.memory.add(state, action, reward, state_, done)
            ##########################################################

            # only start learning once replay memory reaches REPLAY_START_SIZE
            if agent.memory.mem_count > REPLAY_START_SIZE:
                agent.learn()
            
            state = state_
            episode_batch_score += reward
            episode_reward += reward
            
            if done:
                break

        episode_history.append(i)
        episode_reward_history.append(episode_reward)
        episode_reward = 0.0
        
        current_goal_history.append(episode_current_goal)

        # save our model every batches of 100 episodes so we can load later. (note: you can interrupt the training any time and load the latest saved model when testing)
        if i % 100 == 0 and agent.memory.mem_count > REPLAY_START_SIZE:
            torch.save(agent.policy_network.state_dict(), model_file)
            print("average total reward per episode batch since episode ", i, ": ", episode_batch_score/ float(100))
            episode_batch_score = 0
            
            current_time = time.time()
            total_time = current_time - start_time
            print(f'Training took: {total_time/60} minutes!')
            plt.figure()  
            plt.plot(episode_history, episode_reward_history)
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Episode Reward History')
            # Save the plot as an image file
            plt.savefig(episode_reach_goal_plot)

            plt.figure()  
            plt.plot(episode_history, current_goal_history)
            plt.xlabel('Episode')
            plt.ylabel('Reach Goal')
            plt.title('Episode Reach Goal History')
            # Save the plot as an image file
            plt.savefig(episode_reward_plot)
            
        elif agent.memory.mem_count < REPLAY_START_SIZE:
            print(f"waiting for buffer to fill... {i}")
            episode_batch_score = 0

    current_time = time.time()
    total_time = current_time - start_time
    print(f'Training took: {total_time/60} minutes!')
    plt.figure()  
    plt.plot(episode_history, episode_reward_history)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Reward History')
    # Save the plot as an image file
    plt.savefig(episode_reach_goal_plot)

    plt.figure()  
    plt.plot(episode_history, current_goal_history)
    plt.xlabel('Episode')
    plt.ylabel('Reach Goal')
    plt.title('Episode Reach Goal History')
    # Save the plot as an image file
    plt.savefig(episode_reward_plot)

    # Displaying the plot is optional
    # plt.show()

def test_model(model_file):
    env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True)
    agent = DQN_Solver(env)
    agent.policy_network.load_state_dict(torch.load(model_file))
    agent.policy_network.eval()
    average_time = []
    success_rate = []
    start_time = time.time()
    for _ in range(100):
        state = env.reset()[0]
        total_reward = 0
        while True:
            with torch.no_grad():
                q_values = agent.policy_network(torch.tensor(state, dtype=torch.float32))
            action = torch.argmax(q_values).item()
            state, reward, done, info, _ = env.step(action)
            total_reward += reward
            env.render()
            if done:
                average_time.append(start_time)
                start_time = time.time()
                if reward > 300:success_rate.append(1)
                else:success_rate.append(0)
                print(f"Test completed with reward: {total_reward}")
                break
    time.sleep(5)
    env.close()
    
    print("Average time:", sum(average_time) / len(average_time), "seconds")
    print("Average success rate:", sum(success_rate) / len(success_rate))

def load_and_render_simulator():
    env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=True, isDiscrete=True)
    state = env.reset()[0]
    done = False
    while not done:
        _, _, done,_, _ = env.step(env.action_space.sample())  # Take random actions
        env.render()
    env.close()

############################################################################################
## if there is training data available. Check if the model file exists
version = 10
usersName = "PK"
previous_model_file = "trainingPolicy/policy_network_run_around_maze_v" + str(version-1) + "_" + usersName + ".pkl"
model_file = "trainingPolicy/policy_network_run_around_maze_v" + str(version) + "_" + usersName + ".pkl"
episode_reach_goal_plot = "trainingPolicy/episode_reach_goal_plot_v" + str(version) + "_" + usersName + ".png"
episode_reward_plot = "trainingPolicy/episode_reward_plot_v" + str(version) + "_" + usersName + ".png"
if __name__ == "__main__":
    if TRAIN:
        train_model()
    else:
        test_model(model_file)

