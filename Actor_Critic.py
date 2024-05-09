# -*- coding: utf-8 -*-
"""
Actor Network Class
Critic Network Class

@author: Kerbel 2023
"""
import torch # will use pyTorch to handle NN 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical


# 2 Fully Connected Layer Network
class CriticNetwork(nn.Module):
    def __init__(self, input_size, action_size, h1_size, h2_size, h3_size, obs_low, obs_high, device = None):
        super(CriticNetwork, self).__init__()
        
        # Check for device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.obs_low = obs_low
        self.obs_high = obs_high
        self.n_states = input_size

        # Setup critic network layers
        self.fc1 = nn.Linear(input_size + action_size, h1_size)
        self.fc2 = nn.Linear(h1_size, h2_size)
        self.fc3 = nn.Linear(h2_size, h3_size)
        self.value = nn.Linear(h3_size, 1)
        self.bn1 = nn.LayerNorm(h1_size)

    def forward(self, state, actions):
        
        # Check for tensors
        if not torch.is_tensor(state):
          state = torch.from_numpy(state).float().to(self.device)
        if not torch.is_tensor(actions):
          actions = torch.from_numpy(actions).float().to(self.device)
        
        # Normalize the input vector to better map to the output
        state = (state - self.obs_low)/(self.obs_high - self.obs_low)

        # Forward propogation through network with Relu from state
        h = torch.cat([state, actions], dim=-1)  # (B, states+actions)
        state_value = self.fc1(h)
        state_value = self.bn1(state_value)
        state_value = torch.tanh(state_value)        
        state_value = F.elu(self.fc2(state_value))
        state_value = F.elu(self.fc3(state_value))
        value = self.value(state_value)
        return value

class ActorNetwork(nn.Module):
    def __init__(self, input_size, h1_size, h2_size, h3_size, action_size, obs_low, obs_high, device = None):
        super(ActorNetwork, self).__init__()
        
        # Check for device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.n_ca = 1  # Torque
        self.n_da = 1  # Gear change 
        self.n_daa = 3  # gear shifting options: -1, 0, +1   
        self.n_states = input_size
        self.obs_low = obs_low
        self.obs_high = obs_high

        self.fc1 = nn.Linear(input_size, h1_size)
        ### FOR XAVIER INITIALIZATION
        nn.init.xavier_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(h1_size, h2_size)
        ### FOR XAVIER INITIALIZATION
        nn.init.xavier_normal_(self.fc2.weight)
        self.fc3 = nn.Linear(h2_size, h3_size)
        ### FOR XAVIER INITIALIZATION
        nn.init.xavier_normal_(self.fc3.weight)
        
        # Output Nodes
        # Torque is a Gaussian with mean and variance
        self.mean_layer = nn.Linear(h3_size, self.n_ca)
        self.sigma_layer = nn.Linear(h3_size, self.n_ca)
         # There are three gear choices: down, stay, up
        self.action_discrete = nn.Linear(h3_size, self.n_daa)

        self.bn1 = nn.LayerNorm(h1_size)

        
    def forward(self, state):
        """
        Param state is a torch tensor
        Output continuous actions mean and st dev + discrete action probabilites
        """
        if not torch.is_tensor(state):
            state = torch.from_numpy(state).float().to(self.device)  # (num_samples, num_observations)
        if state.ndim == 1:
            state = state.unsqueeze(0)  # (num_samples, num_observations) 


        # Normalize the input vector to better map to the output
        state = (state - self.obs_low)/(self.obs_high - self.obs_low)
        state = state.float()

        # Forward propogation through network with Relu from state
        x = self.fc1(state)
        x = self.bn1(x)
        x = torch.tanh(x)       
        x = self.fc2(x)
        x = F.elu(x)
        x = self.fc3(x)

        B = state.size(0)
        
  
        # Continuous Action Gaussian
        mean = torch.tanh(self.mean_layer(x))  # (B, n_ca)
        sigma_vector = torch.sigmoid(self.sigma_layer(x)) 
        sigma_vector = torch.clamp(sigma_vector,0.000001,float('inf')) # Keep from being 0
        sigma_mat = torch.zeros(size=(B, self.n_ca, self.n_ca), dtype=torch.float32).to(self.device)
        index_loc = torch.arange(0, self.n_ca, 1).to(self.device)
        sigma_mat[:,index_loc,index_loc] = sigma_vector

        # Discrete Action Probabilities
        g_probs = F.softmax(self.action_discrete(x), dim=-1)  # (B, n_da)

        return mean, sigma_mat, g_probs   # (B, n_ca), (B, n_ca), (B, n_da, n_daa_size)

    # Sample Actions
    def select_action(self, state):
        ''' Select continuous and discrete actions from a probability distribution'''
        
        # Forward pass through actor network
        mean, cholesky, d_probs = self.forward(state)
        
        # Continuous
        action_distribution = MultivariateNormal(mean,scale_tril = cholesky)
        action_c = action_distribution.sample()
        action_c = torch.clamp(action_c, -1, 1)  # Restrain sampling to min and max
        logprob_c = action_distribution.log_prob(action_c)

        # Discrete
        dist_d = Categorical(probs = d_probs)
        action_d = dist_d.sample()
        logprob_d = dist_d.log_prob(action_d)

        # Combine actions
        actions = torch.cat([action_c, action_d.unsqueeze(1)],dim=1) # (n_envs, n_actions)
        
        # Return action vector, discrete probabilities, and log probabilities
        return actions, d_probs, logprob_c, logprob_d


    # Get the greedy action choice (mean and highest discrete value)
    def select_greedy_action(self, state):
        
        with torch.no_grad():     
          mean, _, probs_d = self.forward(state) 
          action_c = mean
          action_d = torch.argmax(probs_d, dim = 1)
        
        # Combine action choices
        actions = torch.cat([action_c, action_d.unsqueeze(1)],dim=1)     
           
        # Return action vector, and discrete probabilities for data only   
        return actions, probs_d

