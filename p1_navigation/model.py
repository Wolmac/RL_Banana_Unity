import torch
from torch import nn
import torch.nn.functional as F
# DNN for DQN
class DNN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DNN,self).__init__()
        if state_size < 1 or action_size < 1:
            error('non-valid input')
            del self
            return
        self.state_size = state_size
        self.action_size = action_size
        self.layer_1 = nn.Linear(state_size,2*state_size)
        self.layer_2 = nn.Linear(2*state_size, state_size)
        self.layer_3 = nn.Linear(state_size, state_size)
        self.layer_4 = nn.Linear(state_size, action_size)
    
    def forward(self, inputx):
        # Implementation of a simple DQN network 
        inputx= inputx.float()
        x = F.relu(self.layer_1(inputx))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = self.layer_4(x)
        return x

# DNN for Dueling DQN
class DNN_dueling(nn.Module):
    def __init__(self, state_size, action_size):
        super(DNN_dueling, self).__init__()
        if state_size < 1 or action_size <1:
            # delete the DNN for incorret input
            print('error')
            del self
            return
        self.state_size = state_size
        self.action_size = action_size
        self.layer_1 = nn.Linear(state_size,2*state_size)
        #self.bn_1 = nn.BatchNorm1d(2*state_size)
        self.layer_2 = nn.Linear(2*state_size, state_size)
        #self.bn_2 = nn.BatchNorm1d(state_size)
        self.layer_3 = nn.Linear(state_size, state_size)
        self.layer_4 = nn.Linear(state_size, state_size)
        
        self.layer_5a = nn.Linear(state_size, 2*state_size)
        self.layer_6a = nn.Linear(2*state_size, state_size)
        self.layer_7a = nn.Linear(state_size, action_size)
        
        self.layer_5b = nn.Linear(state_size, 2*state_size)
        self.layer_6b = nn.Linear(2*state_size, state_size)
        self.layer_7b = nn.Linear(state_size, 1)
        

    def forward(self, inputx):
        # Implementation of the dueling loss function proposed by Wang et al. 
        # https://arxiv.org/abs/1511.06581
        inputx = inputx.float()
        x = F.relu(self.layer_1(inputx))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = F.relu(self.layer_4(x))
        v = F.relu(self.layer_5a(x))
        v = F.relu(self.layer_6a(v))
        v = self.layer_7a(v)
        ad = F.relu(self.layer_5b(x))
        ad = F.relu(self.layer_6b(ad))
        ad = self.layer_7b(ad)
        return v + ad - torch.mean(ad, dim = 1, keepdim = True, out = None) 
        

        