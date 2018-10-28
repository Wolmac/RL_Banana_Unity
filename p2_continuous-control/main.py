from unityagents import UnityEnvironment
import numpy as np
import pickle
# select this option to load version 1 (with a single agent) of the environment
#env = UnityEnvironment(file_name='/data/Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64')

# select this option to load version 2 (with 20 agents) of the environment
env = UnityEnvironment(file_name='./Reacher_Linux/Reacher.x86_64')# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])
import random
import torch
#%load_ext autoreload
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
#%matplotlib inline
import importlib
from ddpg_agent import Agent
import ddpg_agent
import model
av_reward = deque(maxlen=100)

importlib.reload(ddpg_agent)
#importlib.reload(model)

agent_list = []
#import ipdb; ipdb.set_trace()
agent_list.append(ddpg_agent.Agent(state_size=33, action_size=action_size, random_seed=100))
for a in range(19):
    agent = ddpg_agent.Agent(state_size=33, action_size=action_size, random_seed=a)
    agent.memory = agent_list[0].memory
    agent_list.append(agent)
num_episodes = 150
max_t = 10000
training_reward_list = []
best_score = 0
for episode in range(num_episodes):
    print(episode)
    env_info = env.reset(train_mode=True)[brain_name]      # reset the environment
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    time_step = 0
    for agent in agent_list:
        agent.reset()
    while True:
        action_list = []
        for ida, agent in enumerate(agent_list):
            action_list.append(agent.act(torch.from_numpy(states[ida,...]).unsqueeze(0)))
        actions = np.asarray(action_list)
        #actions = agent.act(torch.from_numpy(states)) # select an action (for each agent)
        actions = np.asarray(np.clip(actions, -1, 1)).squeeze()                 # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards
        rewards = [0.1 if e> 0.0 else 0.0 for e in rewards]
        dones = env_info.local_done# see if episode finished
        agent_list[0].step(states,actions,rewards,next_states,dones, time_step,agent_list)
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states# roll over states to next time step
        time_step +=1

        if np.any(dones):                                  # exit loop if episode finished
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
    if best_score < np.mean(scores):
        for ida, agent in enumerate(agent_list):
            torch.save(agent.critic_local.state_dict(),'Critic'+str(ida)+'.pth')
            torch.save(agent.actor_local.state_dict(),'Actor'+str(ida)+'.pth')
            best_score = np.mean(scores)
        if np.mean(av_reward)>30:
            print('solved')


    training_reward_list.append(np.mean(scores))
import ipdb; ipdb.set_trace()
with open('rewards.pkl','wb') as f:
    pickle.dump(training_reward_list, f)
