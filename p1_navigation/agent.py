#import the DNN we use (it is just a fully connected one with batchnorm ... makes batchnorm for this env sense ? probably not because the input is well defined (no pixels, just values))
import numpy as np
import random
from collections import namedtuple, deque

# pytorch stuff
import torch
import torch.nn.functional as F
import torch.optim as optim

# set hyperparameters as in the DQN example: globals start with a big letter
Buffer_size = int(1e5)
Batch_size = 64
Gamma = 0.99
Tau = 1e-3 # this is equal to thousand time steps ... sounds good i would say
Lr = 5e-4
Update_every = 4

class ReplayBuffer:
    # Replay buffer with fixed max. length
    def __init__(self, buffer_size, seed, agent):
        # assign an agent to the buffer
        self.agent = agent
        self.action_size = self.agent.action_size
        # define memory with maxlen buffer_size - it contains buffer_size named experience tuples
        self.memory = deque(maxlen=buffer_size)
        # save the td error for each sample in order to implement prioritized experience replay
        self.td_error_memory = deque(maxlen=buffer_size)
        # define batch size
        self.batch_size = self.agent.batch_size
        # save one experience
        self.experience = namedtuple("Experience", field_names = ['state', 'action', 'reward', 'next_state', 'done'])
        # set seed for reproducibility
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done, td_error = 0.1):
        # the nice part is ... an old experience is removed :) automatically
        self.memory.append(self.experience(state, action, reward, next_state, done))
        # add a small constant to the td error - if the td error is 0, nevertheless the sample is used for training
        # the abs is very important!
        # in the beginning, the td error is just noise ... do not use it for priority sampling
        # TODO include td error ... does not work
        self.td_error_memory.append(np.abs(td_error) + 0.1)
    def sample(self):
        # sample according to td error weights
        experiences = random.choices(self.memory, weights= self.td_error_memory , k = self.batch_size)
        # the sampled experience is copied to the specified device
        field_list = []
        for field_name in self.experience._fields:
            if field_name == 'done':
                field_list.append(torch.from_numpy(np.vstack([int(getattr(e, field_name))] for e in experiences if e is not None)).float().to(self.agent.device))

            else:

                field_list.append(torch.from_numpy(np.vstack([getattr(e, field_name)] for e in experiences if e is not None)).float().to(self.agent.device))
        return tuple(field_list)

    def __len__(self):
        # how many samples are already in the memory?
        return len(self.memory)



class Agent():
    # Bot to solve the environment
    def __init__(self, input_size, action_size, seed, dueling = False):
            if dueling:
                from model import DNN_dueling as DNN
            else:
                from model import DNN
            # set seed for reproducibility
            self.seed = random.seed(seed)
            # GPU or CPU ? ... hopefully GPU :)
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            print(self.device)
            # Q networks ... do not forget to assign them to the device!
            self.dqn_local = DNN(input_size, action_size).to(self.device)
            self.dqn_target = DNN(input_size, action_size).to(self.device)
            # use a t_step to update the network only every ,Update_every' step -> samples are more different
            self.t_step = 0
            self.buffer_size = Buffer_size
            self.batch_size = Batch_size
            self.action_size = action_size
            self.input_size = input_size
            # Memory:
            self.memory = ReplayBuffer(Buffer_size, seed, self)
            # the dqn_target is not optimized!
            self.optimizer = optim.Adam(self.dqn_local.parameters(), lr = Lr)


    def step(self, state, action, reward, next_state, done):
            # let networks perform a prediction
            # TODO Td error does not work
            #self.dqn_local.eval()
            #self.dqn_target.eval()
            #with torch.no_grad():
            #    q_local = self.dqn_local(torch.from_numpy(state).float().unsqueeze(0).to(self.device))
            #    q_target = self.dqn_target(torch.from_numpy(next_state).float().unsqueeze(0).to(self.device))
            #    td_error = np.abs((Gamma*(1-int(done))*q_target.max(1)[0] + reward - q_local[..., action]).to('cpu').data.numpy()).squeeze()
            td_error = 0.1

            # set dqn_local in train_mode again
            self.dqn_local.train()
            # save sample with td_error

            self.memory.add(state, action, reward, next_state, done, td_error)
            # perform an update everye Update_every step
            self.t_step = self.t_step % Update_every
            if self.t_step == 0:
                # check if there are enough samples for a batch
                if (len(self.memory)> self.batch_size):
                    # sample from memory
                    experiences = self.memory.sample()
                    # learn
                    self.learn(experiences, Gamma)

    def learn(self, experiences, gamma):
        # get batch of states, ... from experiences
        states, actions, rewards, next_states, dones = experiences
        # use DDQN - with a probability of 20 %, the target is also the local dqn -> double q learning
        if np.random.random() > 0.8:
            Q_max = self.dqn_local(next_states).detach().max(1)[0].unsqueeze(1)
        else:
            Q_max = self.dqn_target(next_states).detach().max(1)[0].unsqueeze(1)
        # compute target ... is dones actually in pytorch ? a tensor ?
        Q_target = rewards + gamma*Q_max*(1-dones)
        Q_local = self.dqn_local(states)
        Q_local = Q_local.gather(1,actions.long())
        loss = F.mse_loss(Q_target, Q_local)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.dqn_local, self.dqn_target, Tau)

    def act(self, state, eps):
        # espilon greedy - with prob eps return random action
        if random.random() < eps:
            return random.randint(0, self.action_size -1)
        # else perform greedy action
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        # set network in evaluation mode
        self.dqn_local.eval()
        with torch.no_grad():
            action_values = self.dqn_local(state)
        # go back in train mode
        self.dqn_local.train()
        # return greedy action
        return np.argmax(action_values.to('cpu').data.numpy())



    def soft_update(self, local_model, target_model, tau):
        # update the parameters of the target network slowly step by step (about 1000 steps if tau is 1e-3)
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(target_param.data *(1-tau) + tau*local_param.data)
