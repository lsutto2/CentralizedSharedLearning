# -*- coding: utf-8 -*-
"""
Group Centralized Policy Class
Group Policy Fitted using Maximum Likelihood

@author: Kerbel 2023
"""

import torch # will use pyTorch to handle NN
from Actor_Critic import CriticNetwork, ActorNetwork
from MemoryClass import Memory
from VehicleModelClass import Vehicle_Model
from DriverOnlyAgent import DriverOnlyAgent
from MPO_Agent import MPO_Agent

import torch.optim as optim
import numpy as np
from torch.distributions import MultivariateNormal, Categorical
import torch.nn.functional as F
import random
from scipy.optimize import minimize
from torch.nn.utils import clip_grad_norm_
import os
import copy
import scipy.io as sio
import time

CURR_DIR = '/scratch/lsutto2/AMT_Central_SubSize/'

def bt(m):
    return m.transpose(dim0=-2, dim1=-1)

def btr(m):
    return m.diagonal(dim1=-2, dim2=-1).sum(-1)
    
class MPO_GroupAgent():
    def __init__(self, group_env, reward_weights, obs_low, obs_high, h1_size = 256, h2_size = 256, h3_size = 256, lr_g = 1e-4, \
                         dual_constraint=0.1, kl_mean_constraint = 0.1, kl_var_constraint = 0.01, \
                         kl_d_constraint=0.005, alpha_c = 10, alpha_d = 10,  \
                         epochs=5, batch_size = 3072, num_batches = 20, iterations = 300,\
                         num_sampled_actions = 30, tau_a = .95, eps_c = .01, eps_d = .01,\
                         grad_norm = .1, cycle_num = 0, device = None, share = 'True', max_num_agents = 5,\
                         beta = 1, r_zeta_c = 1, r_zeta_d = 1, use_adv = False):
      
      # Get or create device 
      if device is None:
          self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      else:
          self.device = device
                 
      self.n_states = group_env.observation_space.shape[0]
      self.env = group_env  # environement to evaluate the group policy    
      self.n_actions = 4   # 1 continuous + 3 discrete options    
      self.share = share
      self.max_agents = max_num_agents
      self.min_agents = max_num_agents

      self.num_g_iterations = iterations
      self.beta = beta
      self.local_zeta_c = r_zeta_c
      self.local_zeta_d = r_zeta_d
      self.g_ε_kl_μ = eps_c
      self.g_epsilon_kl = eps_d
      self.use_advantage = use_adv
      group_lr = lr_g

      # Velocity, Accel, Des Accel, Gear, Prev Ades, Prev Gear, Mass
      obs_low = torch.tensor(obs_low).to(self.device)
      obs_high = torch.tensor(obs_high).to(self.device)
      self.n_ca = 1
      self.n_da = 1
      self.n_daa = 3  # 3 choices for gear [-1, 0, 1]


      self.group_policy = ActorNetwork(self.n_states, h1_size, h2_size, h3_size, self.n_ca, obs_low, obs_high, self.device).to(self.device)
      self.group_policy_target = ActorNetwork(self.n_states, h1_size, h2_size, h3_size, self.n_ca, obs_low, obs_high, self.device).to(self.device)
      # Initialize optimizers for actor and critic
      self.group_policy_optimizer = optim.Adam(self.group_policy.parameters(), lr=group_lr)
      # Set old policy parameters
      self.group_policy_target.load_state_dict(self.group_policy.state_dict())
      
      self.num_batches = num_batches
      self.epochs = epochs
      self.learn_steps_taken = 0 
      self.lam_next = 0
      self.tau_actor = tau_a
      self.grad_norm = grad_norm

      # Continuous Parameters

      self.ε_dual = dual_constraint
      self.ε_kl_μ = kl_mean_constraint  # hard constraint for the KL
      self.ε_kl_Σ = kl_var_constraint  # hard constraint for the KL
      self.η = np.random.rand()
      self.eta_kl_µ = 0.0
      self.eta_kl_S = 0.0
      self.eta_kl_Σ = 0.0

      # Discrete Parameters
      self.alpha_d = alpha_d
      self.epsilon = dual_constraint 
      self.epsilon_kl = kl_d_constraint
      self.eta = np.random.rand() # this eta is for the dual function, we may not need it
      self.eta_kl = 0.0
      self.A_eye = torch.eye(self.n_da).to(self.device)
      
      # Save information for each run
      self.run_info = {
          'actor_h_size' : h1_size,
          'critic_h_size' : h1_size,
          'epochs': epochs, 
          'reward_weights' : reward_weights,  
          'eps_clip' : self.grad_norm,
          'tau_actor' : self.tau_actor,
          'cycle_num ' : cycle_num, 
          'num_sampled_actions' : num_sampled_actions, 
          'share' : self.share, 
          'num_g_iterations': self.num_g_iterations,
          'beta' : self.beta, 
          'local_zeta_c' : self.local_zeta_c,
          'local_zeta_d' : self.local_zeta_d, 
          'g_eps_kl_d' : self.g_ε_kl_μ,
          'g_eps_kl_c' : self.g_epsilon_kl, 
          'num_batches' : self.num_batches, 
          'max_num_agents' : max_num_agents, 
          'batch_size' : batch_size
      }

      # Training Parameters
      self.batch_size = batch_size 
      self.epochs = epochs # number of times the learning process is ran per batch   
      self.num_batchs = num_batches # how many batches to run
      self.num_sampled_actions = num_sampled_actions

      self.torque_high = 20000
      self.torque_low = -20000 

      
      # loss tracking
      self.mean_policy = []
      self.mean_loss_p = [] 
      self.final_eta_kl_c = []
      self.final_eta_kl_d = []
      self.final_eta_kl_sig = []
      self.episode_tracker = []
      self.max_advantage = []
      self.mean_advantage = []
      self.min_advantage = []
      self.final_kl_d_f = []
      self.final_kl_c_f = []
      self.action_low = -1
      self.action_high = 1

      self.numberOfEpsidoesThingsToStorEval = 10
      self.group_eps_eval_results = np.empty((1000, self.numberOfEpsidoesThingsToStorEval))
      self.total_steps = 0
      self.episode_steps = 0 


    # Evaluate group policy if desired
    def evaluate_group_policy(self, roll_outs = 1, stoch = False, cycle_num = 4):
      
      # counters
      g_count = 0
      g_flutter = 0
      p_gear = 1
      gear_switching = np.zeros([3])
      eval_crash = 0
      total_rewards = 0

      states = self.env.evaluation_drive_cycle(cycle_num)  
      max_episode_steps = 40000
 
      # Vectorized environment for 3 masses
      for i in range(1):
          episode_steps = 0

          self.group_eval_data = np.empty((max_episode_steps, 40)) 
          self.group_eval_data[:] = np.nan
          timeout = False          
          while not timeout:

              with torch.no_grad():
                actions, d_probs = self.group_policy.select_greedy_action(states) 
               
              actions = actions[0].detach().cpu().numpy()
              # Actions for environments should be in vector form
              action_env = np.zeros_like(actions)
              
              action_env[0] = np.clip(actions[0] * self.torque_high, self.torque_low, self.torque_high)
              action_env[1] = actions[1] - 1 # hard coding for gear choice -> [-1 0 1]
                       
              states, reward, timeout, info = self.env.step(action_env)

              if info['terminate']:
                done = True

              
              b_vals = d_probs[0].detach().cpu().numpy()
              # Collect Step Data
              dX = info['dX']
              dV = info['velocity']-info['obj_velocity']
              self.group_eval_data[episode_steps,:] = info['time'], info['dX'], dV, info['velocity'],info['acceleration'], info['position'], info['energy'],\
                info['obj_velocity'], info['obj_position'], info['t_traction'], info['gear'], action_env[0],\
                info['engine_torque'], info['des_accel'], info['brake_torque'], info['eng_spd'], action_env[1],  info['w_in_gear'], info['g_control'],\
                reward, info['R1'], info['R2'], info['R3'], info['R4'], info['Rg'], info['mass'], info['crash'], 1, info['R5'], info['R6'], info['cycle_time'],b_vals[0], b_vals[1], b_vals[2], info['eval_drive_cycle'], 0, 0, 0, 0, 0
                  
              # check for fluttering if contains a 1 and a -1 within last 3 steps  (t-2, t-1, t)
              gear_switching = np.array([gear_switching[1], gear_switching[2], info['gear'] - p_gear])  # gear attained - prev gear
                       
              if np.array_equal(np.abs(gear_switching),np.array([0, 1 ,1])) or np.array_equal(np.abs(gear_switching),np.array([1, 1 ,1])):
                  g_flutter += 1 
                  
              g_count += abs(info['gear'] - p_gear)
              p_gear = info['gear']
              eval_crash += info['crash']
              episode_steps +=1 
              total_rewards += reward
 
          self.group_eval_data = self.group_eval_data[:episode_steps, :]
          d_accel = np.subtract(self.group_eval_data[:episode_steps - 1,4],self.group_eval_data[:episode_steps - 1,13])
          a_rsme = np.sqrt(np.square(d_accel).mean())  
                                    # Only need final numbers
          mpgs = info['mpg']
          masses = info['mass'] 
          final_pos = int(info['position'])
          reward_per_step = total_rewards/episode_steps
          shifts_per_m = g_count / max(1,final_pos)
             
         
      return mpgs, episode_steps, eval_crash, a_rsme, g_count, g_flutter, final_pos , total_rewards, reward_per_step, shifts_per_m
                
    def _discrete_kl(self, p1, p2):
        p1 = torch.clamp_min(p1, 0.0001)
        p2 = torch.clamp_min(p2, 0.0001)
        return torch.mean((p1 * torch.log(p1 / p2)).sum(dim=-1))

    def gaussian_kl(self, μi, μ, Ai, A):   # FIXED SIGMA
        """
        decoupled KL between two multivariate gaussian distribution
        C_μ = is the KL divergence of the means
        C_Σ = is the KL divergence of the covariance matrices
        The capital sigmas (Σ) in the above equation are covariance matrices. Which means the upper triangular 
        half should be populated.
        https://wikimedia.org/api/rest_v1/media/math/render/svg/f320aaab05e10a1b6bc2c869081caf70d1f3149b
        :param μi: (B, n)
        :param μ: (B, n)
        :param Ai: (B, n, n)
        :param A: (B, n, n)
        :return: C_μ, C_Σ: mean and covariance terms of the KL
        """
        n = A.size(-1)
        μi = μi.unsqueeze(-1)  # (B, n, 1)
        μ = μ.unsqueeze(-1)  # (B, n, 1)
        # Specific transposes because these are 2 dimensional matrices
        Σi = Ai @ bt(Ai)  # (B, n, n)
        Σ = A @ bt(A)  # (B, n, n)
        Σi_inv = Σi.inverse()  # (B, n, n)
        Σ_inv = Σ.inverse()  # (B, n, n)
        inner_μ = ((μ - μi).transpose(-2, -1) @ Σ_inv @ (μ - μi)).squeeze()  # (B,)
        inner_Σ = torch.log(Σ.det() / Σi.det()) - n + btr(Σ_inv @ Σi)  # (B,)
        C_μ = 0.5 * torch.mean(inner_μ)
        C_Σ = 0.5 * torch.mean(inner_Σ)
        return C_μ, C_Σ
    
    # Call to update/Learn the group centralized policy
    def train_group_policy(self, data_num, episode):
        
        mean_policy, mean_loss_p, kl_μ, kl_Σ , kl, loss_p_c, loss_p_d, max_advantage, mean_advantage, min_advantage = self.learn_group_policy(data_num)
        
        print(mean_policy, mean_loss_p, kl_μ, kl_Σ , kl, loss_p_c, loss_p_d, max_advantage, mean_advantage, min_advantage)
        self.mean_policy.append(mean_policy)
        self.mean_loss_p.append(mean_loss_p) 
        self.final_eta_kl_c.append(kl_μ)
        self.final_eta_kl_d.append(kl)
        self.final_eta_kl_sig.append(kl_Σ)
        self.episode_tracker.append(episode)
        self.final_kl_d_f.append(loss_p_d)
        self.final_kl_c_f.append(loss_p_c)
        self.max_advantage.append(max_advantage)
        self.mean_advantage.append(mean_advantage)
        self.min_advantage.append(min_advantage)

        # Save learning data               
        loss_tracking =  {
                      'mean_policy': self.mean_policy,
                      'mean_loss_p': self.mean_loss_p,
                      'eta_kl_c': self.final_eta_kl_c,
                      'eta_kl_sig':self.final_eta_kl_sig,
                      'eta_kl_d': self.final_eta_kl_d,
                      'episode_tracker': self.episode_tracker, 
                      'max_advantage' : self.max_advantage, 
                      'min_advantage' : self.min_advantage, 
                      'mean_advantage' : self.mean_advantage,
                      'f_kl_d' : self.final_kl_d_f, 
                      'f_kl_c' : self.final_kl_c_f,
                      'total_steps' : self.learn_steps_taken}
                      
        self.update_current_model(data_num)
         
        # Evaluate the group policy              
        #mpgs, episode_steps, eval_crash, a_rsme, g_count, g_flutter, final_pos, total_rewards, reward_per_step, shifts_per_m = self.evaluate_group_policy()
        
        #self.group_eps_eval_results = self.group_eps_eval_results.copy()  
        #self.group_eps_eval_results[episode-1, :] = episode, mpgs, eval_crash, a_rsme, g_count, g_flutter, final_pos, total_rewards, reward_per_step, shifts_per_m
        
        # Save Data File
        #currentDir = os.getcwd()
        if (episode % 2) == 0:
            filename = r'AMT_Central_Group_Routes_{}_eps{}.mat'.format(data_num, episode)
            save_path  = r"{}data/gen_{}/{}".format(CURR_DIR,data_num, filename) 
            print(save_path)
                
            sio.savemat(save_path, {'info': self.run_info, 'loss_tracking' : loss_tracking, 'fleet_num' : data_num})
                           
            
    # Update/Learn the group centralized policy
    def learn_group_policy(self, data_num):

      mean_q_loss = 0
      mean_policy = 0
      mean_loss_p = 0
      actorLearn = 0
      
      self.learn_steps_taken += 1      
      min_advantage = 0
      max_advantage = 0
      mean_advantage = 0
      
      self.η_kl_μ = 0.0
      self.η_kl_Σ = 0.0
      self.eta_kl = 0.0
       
      M = self.num_sampled_actions
      B = self.batch_size
      ds = self.n_states

      agent_policy_list = []
      all_agent_batches = []
      
      # Load most updated fleet models  
      path = r'{}agent_models/gen_{}'.format(CURR_DIR,data_num)
      if not os.path.exists(path):
        os.mkdir(path)
      agent_files = os.listdir(path)
      num_agents = len(agent_files)

      # Checking if the list is empty or not
      if num_agents == 0:
          print("Empty directory")
          # Don't share
          share_kl = False
      else:
          share_kl = True
      
      # Only selecting max_num_agents for the team size
      #num_group = np.random.randint(self.min_agents, self.max_agents + 1, 1)  # Not randomly selecting size
      num_group = self.max_agents
      if share_kl:
            if num_agents < num_group:
                    self.agent_files = random.sample(os.listdir(path), num_agents)  # Are only num_agents available
            else: 
                    self.agent_files = random.sample(os.listdir(path), num_group)  # Randomly choose self.max_agents
            for j, f in enumerate(agent_files): 
                critic_m, actor_m, batches_m = self.load_fleet_models(f, data_num)
                if critic_m == 0: continue
                agent_policy_list.append([critic_m, actor_m])
                all_agent_batches.append(batches_m)
            
            if len(agent_policy_list) == 0: 
                  share_kl = False
            else:   # Don't try to use empty policies
                  share_kl = True

      num_agents = len(agent_policy_list)  # How many agents properly loaded
      self.loss_tracker = [1]
      loss_policy = 0

      eps = torch.tensor(1e-6).to(self.device)
      max_tensor = torch.ones((B)).to(self.device) * 3   

      # return empty if there are no policies
      if not share_kl:   
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0  

      for b in range(self.num_batches):
          
          # Reset with each batch
          agent_states_m = torch.zeros([num_agents, B, ds], device = self.device)
          mean_mj = torch.zeros([num_agents, B, self.n_ca], device = self.device)
          sigma_mj = torch.zeros([num_agents, B, self.n_ca, self.n_ca], device = self.device)
          probs_mj = torch.zeros([num_agents, B, self.n_daa], device = self.device)
          probs_m = torch.zeros([num_agents, self.n_daa, B], device = self.device)
          sigma_m = torch.zeros([num_agents, B], device = self.device)
          advantage = torch.zeros([num_agents, B], device = self.device)
          # Solve for batch non-changing variables
          for i, m in enumerate(agent_policy_list):
                
                # Regress over each agent's state distribution
                agent_states_m[i, :, :] = torch.from_numpy(all_agent_batches[i][b, :, :]).float().to(self.device) 
                

                with torch.no_grad():
                    mean_mj[i, :, :], sigma_mj[i, :, :], probs_mj[i, :, :] = m[1].forward(agent_states_m[i, :, :]) 
                    sigma_m[i, :] = sigma_mj[i, :, 0, 0]
                    
                    # Advantage Weighted Regression 
                    if self.use_advantage:
                        # Sample actions from entire distribution for V(s)
                        # Continuous
                        mean_all = torch.zeros([B,1]).to(self.device)
                        sigma_all = torch.ones([B,1,1]).to(self.device) * .8
                        cont_dist_t = MultivariateNormal(mean_all,scale_tril = sigma_all)
                        sampled_actions_c = torch.clamp(cont_dist_t.sample((M,)),min = self.action_low,max = self.action_high) # (M, B, n_ca)
                        # Discrete
                        probs_all = torch.ones([B, 3]).to(self.device) * 1/3
                        pi_t = Categorical(probs=probs_all)
                        actions_d = pi_t.sample((M,)) # (M, B*R) The discrete action sampling
                        expanded_actions_d = F.one_hot(actions_d,num_classes=self.n_daa) # (M, B, n_daa)
                        sampled_actions = torch.cat([sampled_actions_c,expanded_actions_d],dim=2) # (M, B, n_daa + n_ca)
                        
                        expanded_states = agent_states_m[i][None, :, :].expand(M, -1, -1)  # (M, B, ds)
                        
                        # Sampled Q-value (over all actions)
                        q_value = m[0](expanded_states.reshape(-1,ds), \
                                sampled_actions.reshape(-1,self.n_actions)).reshape(M,B) # (M,B)
                        # V(s) as the mean of the sampled Q-values
                        v_value = torch.mean(q_value, dim = 0) # (B)
                        
                        # Q-values for current policies (greedy actions)
                        actions_d_greedy = torch.argmax(probs_mj[i, :, :], dim = 1)
                        expanded_greedy_actions = F.one_hot(actions_d_greedy, num_classes = self.n_daa)
                        actions_star = torch.cat([mean_mj[i, :, :], expanded_greedy_actions], dim = 1)  # (B, n_ca + n_daa)
                        q_value_star = m[0](agent_states_m[i, :, :], actions_star).squeeze() 
                        
                        # Advantage function Q(s, a) - V(s) / beta
                        a_value = torch.min((q_value_star - v_value)/self.beta, max_tensor)
                        advantage[i] = torch.exp(a_value)  # (B)
                    else:
                        advantage[i] = 1
                    
                    # Transposed for regression below
                    probs_m[i] = torch.transpose(probs_mj[i, :, :], 0, 1)

                    # For constraint from previous group policy (only using states from last agent)
                    mean_t,chole_t, probs_t = self.group_policy_target.forward(agent_states_m[i, :, :])
 
          li = 0 
          max_advantage = torch.zeros([num_agents])
          min_advantage = torch.ones([num_agents])
          mean_advantage = torch.zeros([num_agents])                   
          while (li < self.num_g_iterations): 
                loss_p = 0
                loss_policy = 0
                for i, m in enumerate(agent_policy_list):
                    # Get group policy for the batch
                    mean,chole, probs = self.group_policy.forward(agent_states_m[i, :, :])
                    pi = Categorical(probs=probs)

                    # regressing over probabilities of all discrete action choices
                    all_actions_d = torch.arange(self.n_daa)[..., None].expand(self.n_daa, B).to(self.device)  # (da,B)
                    logprobs_d = probs_m[i] * pi.expand((self.n_daa, B)).log_prob(all_actions_d)  #(n_daa, B)
                    
                    logprobs_d = torch.sum(logprobs_d, dim=0)   # (B)
                    
                    # Continuous using agent mean and sigma
                    sigma = chole[:, 0, 0]
                    logprobs_c = torch.log(torch.max(eps, sigma)) + (sigma_m[i]**2 + (mean_mj[i, :, :] - mean)**2)/(2*torch.max(eps, sigma)**2)
                    
                    loss_p_d = -torch.mean(logprobs_d * advantage[i] * self.local_zeta_d)   # Always negative -- want a larger negative value
                    loss_p_c = torch.mean(advantage[i] * logprobs_c * self.local_zeta_c)
                    
                    max_advantage[i] = torch.max(advantage[i])
                    mean_advantage[i] = torch.mean(advantage[i])
                    min_advantage[i] = torch.min(advantage[i])
                 
                    
                    loss_p +=  loss_p_d + loss_p_c
                               
                loss_policy = loss_p / num_agents
                
                
                # Fixed KL discrete (target | present)
                kl = self._discrete_kl(p1=probs_t, p2=probs)    # FIXED
                kl_μ, kl_Σ = self.gaussian_kl(μi=mean_t, μ=mean, \
                                         Ai=chole_t, A=chole)
 
                # Update lagrange multipliers by gradient descent
                # KL Divergence (Target, Current)
                self.eta_kl -= self.alpha_d * (self.g_epsilon_kl - kl).detach().item()
                self.eta_kl_μ -= self.alpha_c_μ * (self.g_ε_kl_μ - kl_μ).detach().item()
                self.eta_kl_Σ -= self.alpha_c_Σ * (self.ε_kl_Σ - kl_Σ).detach().item()
          
                if self.eta_kl_μ < 0.0:
                    self.eta_kl_μ = 0.0
                if self.eta_kl_Σ < 0.0:
                    self.eta_kl_Σ = 0.0
                if self.eta_kl < 0.0:
                    self.eta_kl = 0.0
                
                loss_policy -=  ( self.eta_kl * (self.g_epsilon_kl  - kl) \
                                + self.eta_kl_μ * (self.g_ε_kl_μ - kl_μ) \
                                + self.eta_kl_Σ * (self.ε_kl_Σ - kl_Σ))   
                    
                self.group_policy_optimizer.zero_grad()
                loss_policy.backward() 
                clip_grad_norm_(self.group_policy.parameters(), self.grad_norm)
                self.group_policy_optimizer.step() 
                mean_policy += loss_policy
                mean_loss_p += loss_p
                actorLearn += 1
                li += 1
 
      #print(f'**GROUP POLICY Loss P {loss_policy:.4e} mean_loss_p  {torch.mean(mean_loss_p)}  mean_policy  {loss_p_c} : {loss_p_d}  ')
      self.update_actor_target_network(self.tau_actor)

      
      return mean_policy.item()/actorLearn, mean_loss_p.item() / actorLearn, kl_μ.item(), kl_Σ.item() , kl.item(), loss_p_c.item(), loss_p_d.item(), max_advantage.cpu().numpy(), mean_advantage.cpu().numpy(), min_advantage.cpu().numpy()


          
    def update_actor_target_network(self,tau = 1):
      # update target networks 
      #print('update actor')
      for target_param, param in zip(self.group_policy_target.parameters(), self.group_policy.parameters()):
          target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    
    def save_group_model(self, episode_id, data_num):
        data = {
            'actor_state_dict': self.group_policy.state_dict(),
            'target_actor_state_dict': self.group_policy_target.state_dict(),
            'actor_optim_state_dict': self.group_policy_optimizer.state_dict(),
            'group_eps_eval_results': self.group_eps_eval_results 
        }

        filename = 'AMT_GroupModel_{}.pt'.format(data_num)
        save_path  = r"{}/group_models/gen_{}/{}".format(CURR_DIR, data_num, filename)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        torch.save(data, save_path)
        
    def update_current_model(self, data_num):
        data = {
              'actor_state_dict': self.group_policy.state_dict()
        }
        
        filename = 'AMT_GroupModel_{}.pt'.format(data_num)
        save_path  = r"{}group_models/gen_{}".format(CURR_DIR, data_num)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_path  = r"{}group_models/gen_{}/{}".format(CURR_DIR, data_num, filename)
        torch.save(data, save_path)
            
    def load_model_policy(self, filename): 
        
        self.save_path  = F"{CURR_DIR}group_models/pre_agent/{filename}"  
        checkpoint = torch.load(self.save_path, map_location=torch.device(self.device))
        self.group_policy.load_state_dict(checkpoint['actor_state_dict'])
        self.group_policy_target.load_state_dict(checkpoint['target_actor_state_dict'])

        
    def load_fleet_models(self, filename, data_num): 

        save_path  = r"{}agent_models/gen_{}/{}".format(CURR_DIR, data_num, filename)
        
        if not os.path.isfile(save_path):
          return 0, 0, 0
        h1_size = h2_size = h3_size = 256
        
        obs_low = np.array([0, -3, -3, 1, -3, 1, 5000])
        obs_high = np.array([38, 3, 3, 10, 3, 10, 25000])
        obs_low = torch.tensor(obs_low).to(self.device)
        obs_high = torch.tensor(obs_high).to(self.device)
        agent_critic = CriticNetwork(self.n_states, self.n_actions, h1_size, h2_size, h3_size, obs_low, obs_high, self.device).to(self.device)
        agent_policy = ActorNetwork(self.n_states, h1_size, h2_size, h3_size, self.n_ca, obs_low, obs_high, self.device).to(self.device)  
        
        for n in range(10):
            
            try:
                checkpoint = torch.load(save_path, map_location=torch.device(self.device))
                agent_states = checkpoint['saved_state_dist']
            except:
                print(f'Attempt # {n}')
                print('Retry Loading Agent')
                print(save_path)
                time.sleep(n)
            else: 
                agent_critic.load_state_dict(checkpoint['critic_state_dict'])
                agent_policy.load_state_dict(checkpoint['actor_state_dict'])
                return agent_critic, agent_policy, agent_states
        return 0, 0, 0
 
