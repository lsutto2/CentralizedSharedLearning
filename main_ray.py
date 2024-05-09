# -*- coding: utf-8 -*-
"""
Run file for fleet simulation using Ray
Centralized Shared Policy Learning

Option for max team size to learn group policy
20 Vehicles
Suburban routes

@author: LKerbel 2023
"""

import sys
from DeterministicTrackClass import DeterministicTrack
from MPO_Agent import MPO_Agent
from MPO_GroupAgent import MPO_GroupAgent
import scipy.io as sio
import numpy as np
import gym
import ray
import logging
import os
import asyncio
import time 
import sys


# Set how much of the gpu each instance requires
@ray.remote(num_gpus=.15)
class LoadAgentAndEnvironment():
    def __init__(self, agent_num):
        print(f'loaded class for num {agent_num}')
        self.agent_num = agent_num
        
        print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
        print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
        
        # Data index
        self.data_num = 6104
        # Enable fleet sharing
        self.share = True
        #Set reward weights: Wa, Wtt, gamma, Wg, Wp, Wgp, Wflutter
        self.reward_weights = [0.6100, 0.0800, 0.2700, 0.00010, 0.0050, 0.001, 0.5]

        # Load independent environment for each vehicle
        seed = np.random.randint(0, 100)
        self.env =  DeterministicTrack(self.reward_weights, seed = seed)
        self.env.seed(seed)
        
    # Load the RL agent for the vehicle
    def load_actor(self):

        # Fleet sharing weights
        zeta_d_r = .2
        zeta_c_r = .1
        zeta_s_r = .01
 
        # Set memory buffer size for the agent
        memory_size = 300000
        
        # State min and max
        obs_low = np.array([0, -3, -3, 1, -3, 1, 5000])
        obs_high = np.array([38, 3, 3, 10, 3, 10, 25000])
        
        # Create a RL agent
        self.agent = MPO_Agent(self.env, self.reward_weights, obs_low, obs_high, h1_size = 256, h2_size = 256, h3_size = 256, lr_a = 1e-4, lr_c = 5e-4, \
                         gamma = .97, dual_constraint=0.1, kl_mean_constraint = 0.1, kl_var_constraint = 0.01, \
                         kl_d_constraint=0.01, alpha_c = 10, alpha_d = 10,  \
                         lam = 1, sample_steps=4000, lagrange_it=10, \
                         epochs=10, batch_size = 2048, num_batches = 25, retrace_steps = 6, \
                         num_sampled_actions = 30, memory_size = memory_size, tau_a = .95, tau_c = .01,\
                         grad_norm = .1, cycle_num = 0, device = None, share = self.share, local_zeta_c = zeta_c_r, local_zeta_d = zeta_d_r)
        
        # Number the agent (for data and file saving)                 
        self.agent.number = self.agent_num

        # Preload a policy if desired
        self.agent.load_model_policy('AMT_agent1_RL_5.10.pt')

    # Create a group centralized policy        
    def load_group_agent(self):
        # State min and max
        obs_low = np.array([0, -3, -3, 1, -3, 1, 5000])
        obs_high = np.array([38, 3, 3, 10, 3, 10, 25000])

        # Group learning hyperparameters
        group_lr = 5e-5
        beta = .6
        zeta_d_r = 1
        zeta_c_r = 1
        
        # Set the maximum team size to learn the group policy
        # Randomly selects max_num_agents from the whole fleet
        max_num_agents = 10
        
        # Create group agent policy
        self.group_agent = MPO_GroupAgent(self.env, self.reward_weights, obs_low, obs_high, h1_size = 256, h2_size = 256, h3_size = 256, lr_g = group_lr,\
                 dual_constraint=0.1, kl_mean_constraint = 0.1, kl_var_constraint = 0.01, \
                 kl_d_constraint=0.05, alpha_c = 10, alpha_d = 10,  \
                 epochs=10, batch_size = 2048, num_batches = 25, iterations = 350,\
                 num_sampled_actions = 30, tau_a = .95, eps_c = .01, eps_d = .01,\
                 grad_norm = .1, cycle_num = 0, device = None, share = self.share, max_num_agents = max_num_agents,\
                 beta = beta, r_zeta_c = zeta_c_r, r_zeta_d = zeta_d_r, use_adv = True)
       
        # Preload a policy if desired
        self.group_agent.load_model_policy('AMT_agent1_RL_5.10.pt')
        
    def train_agents(self, episode):
        print(f'********* Training agent {self.agent.number}')
        
        eval_every_n_steps = 2e4 # Stop learning and evaluate on a deterministic environment
        # Learn for an episode and evaluate
        self.agent.run_training(eval_every_n_steps, self.data_num, episode)
    
    def train_group_agent(self, episode):
        print(f'********* Training Group Agent')
        # Learn/Update the group central policy
        self.group_agent.train_group_policy(self.data_num, episode)
      
        
if __name__ == "__main__":

      # Number of vehicles in the fleet
      num_agents = 20
      num_episodes = 100  # Max number of training episodes
      
      # Setup Ray.io
      ip_head = sys.argv[1]
      ray.shutdown()
      ray.init(address=ip_head)
   
      # Start all instances (vehicles and group policy) 
      gpu_actors = [LoadAgentAndEnvironment.remote(i) for i in range(num_agents)] 
      group_actor = LoadAgentAndEnvironment.remote(num_agents + 1)
      
      # Load the actors for each vehicle
      result_ids = ray.get([m.load_actor.remote() for m in gpu_actors])  
      time.sleep(60)
      # Load the group policy
      group_id = ray.get(group_actor.load_group_agent.remote())
          
      # Traing each vehicle for an episode, then learn the group policy           
      for j in range(num_episodes):
          print(f'Episode {j}')
          start = time.time()
          episode = ray.put(j+1)
          result_ids = [m.train_agents.remote(episode) for m in gpu_actors]
          while len(result_ids):
              done_id, result_ids = ray.wait(result_ids)
              ray.get(done_id[0])
          group_id = ray.get(group_actor.train_group_agent.remote(episode))
          print("Finished episode: duration =", time.time() - start)

              
      # End of Simulation        
      print('FINISHED ALL EPISODES')
      ray.shutdown()     