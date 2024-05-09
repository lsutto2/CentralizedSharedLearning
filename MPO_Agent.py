# -*- coding: utf-8 -*-
"""
RL Agent Class Using Maximum a Posteriori Policy Optimisation (MPO)

Algorithms from https://arxiv.org/abs/1806.06920

@author: Kerbel 2023
"""

import torch # will use pyTorch to handle NN
from Actor_Critic import CriticNetwork, ActorNetwork
from MemoryClass import Memory
from VehicleModelClass import Vehicle_Model
from DriverOnlyAgent import DriverOnlyAgent

import torch.optim as optim
import numpy as np
from torch.distributions import MultivariateNormal, Categorical
import torch.nn.functional as F
from scipy.optimize import minimize
from torch.nn.utils import clip_grad_norm_
import os
import copy
import scipy.io as sio

CURR_DIR = '/scratch/lsutto2/AMT_Central_SubSize/'

def bt(m):
    return m.transpose(dim0=-2, dim1=-1)

def btr(m):
    return m.diagonal(dim1=-2, dim2=-1).sum(-1)

class MPO_Agent():
    def __init__(self, env, reward_weights, obs_low, obs_high, h1_size = 256, h2_size = 256, h3_size = 256, lr_a = 5e-5, lr_c = 1e-4, \
                         gamma = .95, dual_constraint=0.1, kl_mean_constraint = 0.1, kl_var_constraint = 0.01, \
                         kl_d_constraint=0.005, alpha_c = 10, alpha_d = 10,  \
                         lam = 1, sample_steps=4000, lagrange_it=10, \
                         epochs=5, batch_size = 3072, num_batches = 20, retrace_steps = 6, \
                         num_sampled_actions = 30, memory_size = 300000, tau_a = .95, tau_c = .01,\
                         grad_norm = .1, cycle_num = 0, device = None, share = True, local_zeta_c = 1, local_zeta_d = 1):
      
      # Set or create current device
      if device is None:
          self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      else:
          self.device = device
          
      self.zeta_cont = local_zeta_c
      self.zeta_disc = local_zeta_d
      self.share = share
      self.number = 0     # This gets set outside the class       
      self.env = env
      self.n_states = len(obs_low)
      self.n_envs = 1  # RIGHT NOW ONLY 1 ENVIRONMENT PER AGENT (for memory buffer)
      self.n_actions = 4   # 1 continuous + 3 discrete options

      self.gamma = gamma
      self.epochs = epochs
      self.lam = lam
      self.step_track = -sample_steps*3 
      self.sample_steps = sample_steps
      self.learn_steps_taken = 0 
      self.tau_actor = tau_a
      self.tau_critic = tau_c
      self.grad_norm = grad_norm

      # Continuous Parameters
      self.n_ca = 1
      self.alpha_c_μ = alpha_c
      self.alpha_c_Σ = alpha_c
      self.ε_dual = dual_constraint
      self.ε_kl_μ = kl_mean_constraint  # hard constraint for the KL
      self.ε_kl_Σ = kl_var_constraint  # hard constraint for the KL

      # Discrete Parameters
      self.n_da = 1
      self.n_daa = 3  # 3 choices for gear [-1, 0, 1]
      self.alpha_d = alpha_d
      self.epsilon = dual_constraint 
      self.epsilon_kl = kl_d_constraint
      
      obs_low = torch.tensor(obs_low).to(self.device)
      obs_high = torch.tensor(obs_high).to(self.device)


      # Save information for each run
      self.run_info = {
          'critic_lr': lr_c,
          'actor_lr' : lr_a,
          'sample_steps' : self.sample_steps,
          'gamma' : gamma,
          'lambda' : lam,
          'actor_h_size' : h1_size,
          'critic_h_size' : h1_size,
          'epochs': epochs, 
          'reward_weights' : reward_weights, 
          'dual_constrain_ct' : self.ε_dual,
          'kl_mean' : kl_mean_constraint, 
          'kl_var' : kl_var_constraint,
          'alpha_c' : alpha_c,
          'alpha_d' : alpha_d,
          'dual_constraint_d' : self.epsilon, 
          'eps_kl_d': self.epsilon_kl, 
          'retrace_steps' : retrace_steps, 
          'eps_clip' : self.grad_norm,
          'tau_actor' : self.tau_actor,
          'tau_critic' : self.tau_critic, 
          'cycle_num ' : cycle_num, 
          'zeta_cont' : self.zeta_cont,
          'zeta_disc' : self.zeta_disc,
          'share' : self.share
      }
      
      
      # Setup network - Actor and Critic are the same size/layers
      self.critic = CriticNetwork(self.n_states, self.n_actions, h1_size, h2_size, h3_size, obs_low, obs_high, self.device).to(self.device)
      self.critic_target = CriticNetwork(self.n_states, self.n_actions, h1_size, h2_size, h3_size, obs_low, obs_high, self.device).to(self.device)
      self.policy = ActorNetwork(self.n_states, h1_size, h2_size, h3_size, self.n_ca, obs_low, obs_high, self.device).to(self.device)
      self.policy_target = ActorNetwork(self.n_states, h1_size, h2_size, h3_size, self.n_ca, obs_low, obs_high, self.device).to(self.device)
      # Initialize optimizers for actor and critic
      self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_c)
      self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr_a)
      # Set old policy parameters
      self.critic_target.load_state_dict(self.critic.state_dict())
      self.policy_target.load_state_dict(self.policy.state_dict())
      self.group_policy = ActorNetwork(self.n_states, h1_size, h2_size, h3_size, self.n_ca, obs_low, obs_high, self.device).to(self.device)
      

      # Training Parameters
      self.rollouts = 1 # number of greedy episode evaluations to perform
      self.n_steps = sample_steps # how many steps are taken before learning
      self.lagrange_it = lagrange_it # how many times the actor is updated
      self.batch_size = batch_size 
      self.epochs = epochs # number of times the learning process is ran per batch   
      self.num_batchs = num_batches # how many batches to run
      self.retrace_steps = retrace_steps     
      self.num_sampled_actions = num_sampled_actions

      # Discrete
      self.eta = np.random.rand() # this eta is for the dual function, we may not need it
      self.eta_kl = 0.0
      self.A_eye = torch.eye(self.n_da).to(self.device)

      # Continuous
      self.η = np.random.rand()
      self.η_kl_μ = 0.0
      self.η_kl_Σ = 0.0
      self.torque_high = 20000
      self.torque_low = -20000 

      # Replay buffer / Memory (independent from fleet)
      self.memory = Memory(memory_size, self.n_states, self.n_ca + self.n_da,1)
      
      # Initial Placeholders
      self.critic_loss = 0
      self.policy_loss = 0
      self.total_steps = 0
      
      # loss tracking
      self.mean_q_loss = []
      self.mean_policy = []
      self.mean_loss_p = []
      self.final_eta_kl_c = []
      self.final_eta_kl_d = []
      self.final_eta_kl_sig = []
      self.final_kl_d_f = []
      self.final_kl_c_f = []
      self.final_loss_p_r = []
      self.episode_tracker = []
      self.learn_steps_taken = 0 
      self.min_q_val = []
      self.max_q_val = []
      self.mean_q_val = []
      
      self.action_low = -1
      self.action_high = 1
      
      self.numberOfEpisodeThingsToStore =16
      self.numberOfEpsidoesThingsToStorEval = 16
      self.eps_train_results = np.empty((1000, self.numberOfEpisodeThingsToStore))
      self.eps_eval_results = np.empty((1000, self.numberOfEpsidoesThingsToStorEval))
      self.total_steps = 0
      self.episode_steps = 0

    # KL divergence for categorical probability distributions
    def _discrete_kl(self, p1, p2):
        p1 = torch.clamp_min(p1, 0.0001)
        p2 = torch.clamp_min(p2, 0.0001)
        return torch.mean((p1 * torch.log(p1 / p2)).sum(dim=-1))
    
    # KL divergence for gaussian distributions
    def gaussian_kl(self, μi, μ, Ai, A):   
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

   # Calculate the target Q value using Q-retrace algorithm (https://arxiv.org/abs/1606.02647v2)
    def calc_retrace_sampling_based(self, state_batch, action_batch, reward_batch, \
                    next_state_batch, done_batch, logprob_c_batch, logprob_d_batch, trunc_batch):
          
          B = self.batch_size
          R = self.retrace_steps
          da = self.n_daa
          ds = self.n_states
          M = self.num_sampled_actions

          # Reshape Batch to Batch * Retrace Steps
          next_state_batch =  next_state_batch.reshape(-1, ds) # (B * R, ds)
          next_state_batch = torch.from_numpy(next_state_batch).float().to(self.device) 
          action_batch = action_batch.reshape(-1,self.n_da +self.n_ca) # (B * R, 2)
          action_batch = torch.from_numpy(action_batch).float().to(self.device)
          # the continuous action goes first
          action_d_batch = action_batch[:,self.n_ca:] # (B*R, n_discrete)
          action_c_batch = action_batch[:,:self.n_ca] # (B*R, n_ca)
          state_batch =  state_batch.reshape(-1, ds) # (B * R, ds)
          logprob_d_batch = logprob_d_batch.reshape(-1,) # (B * R,)
          logprob_d_batch = torch.from_numpy(logprob_d_batch).float()
          logprob_c_batch = logprob_c_batch.reshape(-1,) # (B * R,)
          logprob_c_batch = torch.from_numpy(logprob_c_batch).float()
          
          # Solve for Q retrace
          with torch.no_grad():
 
            #Expected Q(s+,a) per policy
            mean,chole, pi_probs = self.policy.forward(next_state_batch)  # (B * R, da)
            
            # Discrete - Sample M actions            
            pi = Categorical(probs=pi_probs)  # (B * R,)
            cont_dist = MultivariateNormal(mean,scale_tril = chole)
            actions_d = pi.sample((M,)) # (M, B*R, 1) The discrete action sampling
            expanded_actions_d = F.one_hot(actions_d,num_classes=self.n_daa) # (M, B*R, n_da) The discrete action sampling   
                
            # Continuous - Sample M actions           
            actions_c = cont_dist.sample((M,)) # (M, B*R, n_ca)
            actions_c = torch.clamp(actions_c,min = self.action_low, max = self.action_high)
            sampled_actions = torch.cat([actions_c,expanded_actions_d],dim=2) # (M,B*R, n_da + n_ca)
            expanded_next_states = next_state_batch[None, :, :].expand(M, -1, -1)  # (M, B*R, ds)   
            expected_next_q = self.critic_target(expanded_next_states.reshape(-1,ds), \
                                    sampled_actions.reshape(-1,self.n_actions)).reshape(M,B*R).mean(axis = 0) # (B*R)
            expanded_actions = F.one_hot(action_d_batch.squeeze().long(),num_classes=self.n_daa) # (B*R, n_da)
            expanded_actions = torch.cat([action_c_batch, expanded_actions],dim=1) # (B*R, n_da + n_ca)
      
            q_target =  self.critic_target(
                  state_batch, # (B * R, ds),
                  expanded_actions
              ).squeeze(-1)  # (B * R,) 
        
            
            # Calculate probability of action set being taken under the current policy
            mean,chole, probs = self.policy(torch.from_numpy(state_batch).float().to(self.device))
            pi = Categorical(probs=probs)
            dist_c = MultivariateNormal(mean,scale_tril = chole)
            logprobs_d = pi.log_prob(action_d_batch.squeeze())
            logprobs_c = dist_c.log_prob(action_c_batch).squeeze()
            logprobs = logprobs_d + logprobs_c

            # Calculate probability of action set being taken under the policy it was collected
            oldlogprobs = logprob_d_batch + logprob_c_batch
            # Calculate cs
            ratios = (logprobs.cpu() - oldlogprobs).clamp(max=0).exp()
            
            # Change shape back to (B, R)  
            q_target = q_target.reshape(B, R).cpu().numpy()  # (B,R)
            expected_next_q = expected_next_q.reshape(B, R).cpu().numpy()   # (B,R)
            ratios = ratios.reshape(B, R).cpu().numpy()  

            # Calculate Q Retrace for batch
            ck = 1
            Q_o = np.zeros(self.batch_size)  
            Q_o[:] = q_target[:,0]

            # If episode ends... stop Q_o calculations?
            # done_batch == failed, done_batch[:,t] == timeout
            
            # There is always a next Q --- data may be truncated
            ep_done = np.ones(self.batch_size) 
            for t in range(self.retrace_steps):  
                  Q_o[:] += (self.gamma**t*ck*(reward_batch[:,t] + \
                      self.gamma*expected_next_q[:,t] - q_target[:,t]))*ep_done[:]
                  ep_done[:] *= (1-done_batch[:,t]) * (1-trunc_batch[:,t])
                  if t < self.retrace_steps - 1: ck *= self.lam*ratios[:,t+1]

          return Q_o  


    # Update the actor and critic networks (MPO and Q-retrace)
    def learn(self, data_num):

      mean_q_loss = 0
      mean_policy = 0
      mean_loss_p = 0
      actorLearn = 0
      criticLearn = 0
      self.learn_steps_taken += 1
      
      # Setup sharing
      share_local = self.share
      if self.share:
          # Get the current group policy at each step
          self.group_policy, share_local = self.load_group_policy(data_num)

      M = self.num_sampled_actions
      B = self.batch_size
      ds = self.n_states
      
      # Reset eta values at each learn call
      self.eta_kl = 0
      self.η_kl_μ = 0.0
      self.η_kl_Σ = 0.0
      zero_tensor = torch.tensor(0).to(self.device)
      self.saved_state_dist = np.zeros([self.num_batchs, B, ds])
      self.loss_tracker = [1]
       
      # Learn for each batch 
      for b in range(self.num_batchs):

        state_batch, action_batch, reward_batch, next_state_batch, done_batch, logprob_c_batch, logprob_d_batch, is_truncated = self.memory.sample_random_steps(self.batch_size, self.retrace_steps)
        
        # Save state batches to learn group policy
        self.saved_state_dist[b, :, :] = state_batch[:, 0, :]
            
        # Calculate in one step
        Q_retrace_batch = self.calc_retrace_sampling_based(state_batch, action_batch, reward_batch, next_state_batch, done_batch, logprob_c_batch, logprob_d_batch, is_truncated)

        states_old = torch.from_numpy(state_batch[:, 0, :]).float().to(self.device)
        actions_old = torch.from_numpy(action_batch[:,0]).float().to(self.device)
        actions_old_c = actions_old[:,:self.n_ca] # (B, n_ca)
        actions_old_d = actions_old[:,self.n_ca:] # (B, n_discete)

        next_states_old = torch.from_numpy(next_state_batch[:, 0, :]).float().to(self.device)
        expanded_next_states = next_states_old[None, :, :].expand(M, -1, -1)  # (M, B, ds)
        expanded_next_states.reshape(-1, ds)
        
        q_retrace = torch.tensor(Q_retrace_batch, dtype=torch.float).to(self.device)

        for _ in range(self.epochs):

            # Update the critic
            expanded_actions = F.one_hot(actions_old_d.long().squeeze(),num_classes=self.n_daa) # (B*R, n_daa)
            expanded_actions = torch.cat([actions_old_c, expanded_actions],dim=1).float().to(self.device) # (B*R, n_daa + n_ca)
            q_i =  self.critic(
                    states_old, # (B * R, ds),
                    expanded_actions
                ).squeeze(-1)  # (B * R,)
                 

            # q_i is grad, q_retrace is stationary
            q_loss = F.mse_loss(q_i, q_retrace.detach())


            self.critic_optimizer.zero_grad()
            q_loss.backward()
            criticLearn += 1
            clip_grad_norm_(self.critic.parameters(), self.grad_norm)
            self.critic_optimizer.step()  
            mean_q_loss += q_loss.item()

            # Update the actor
            with torch.no_grad():
              # Expected Q(s+,a) per policy
              mean_t,chole_t, probs_t = self.policy_target.forward(states_old)  # (B, da)
              
              # Load group model with batch of current states
              if self.share: 
                  group_mean, group_chole, group_probs = self.group_policy.forward(states_old)
  
              pi_t = Categorical(probs=probs_t)  # (B,)
              cont_dist_t = MultivariateNormal(mean_t,scale_tril = chole_t)
              actions_d = pi_t.sample((M,)) # (M, B*R) The discrete action sampling
              expanded_actions_d = F.one_hot(actions_d,num_classes=self.n_daa) # (M, B, n_da) The discrete action sampling to num of possible actions               
              actions_c = torch.clamp(cont_dist_t.sample((M,)),min = self.action_low,max = self.action_high) # (M, B*R, n_ca)
              sampled_actions = torch.cat([actions_c,expanded_actions_d],dim=2) # (M, B, n_daa + n_ca)
              expanded_states = states_old[None, :, :].expand(M, -1, -1)  # (M, B, ds)
              target_q = self.critic_target(expanded_states.reshape(-1,ds), \
                            sampled_actions.reshape(-1,self.n_actions)).reshape(M,B) # (M,B)
              target_q_np = target_q.cpu().numpy()  # (M, B)
            
            # E-Step
            # Update Dual-function
            def dual(η):
                """
                dual function of the non-parametric variational
                g(η) = η*ε + η \sum \log (\sum \exp(Q(a, s)/η))
                """
                max_q = np.max(target_q_np, 0)
                return η * self.ε_dual + np.mean(max_q) \
                    + η * np.mean(np.log(np.mean(np.exp((target_q_np - max_q) / η), axis=0)))
            
            bounds = [(1e-6, None)]

            # to handle cases when the initial guess of η violates constraints 
            while True:
              try:
                  res = minimize(dual, np.array([self.η]), method='SLSQP', bounds=bounds)
                  break
              except Exception as inst:

                  self.η = np.random.rand()
                  print(f'initial guess of eta was wrong   {self.η}')
            
            self.η = res.x[0]


            target_q = target_q/self.η
            baseline = torch.max(target_q, 0)[0]
            target_q = target_q - baseline
            qij = torch.softmax(target_q, dim=0)  # (M, B) 
            
            # M-step
            # update policy based on lagrangian
            li = 0
            loss_t = .1
            loss_lgr = .1
            

            while (li < self.lagrange_it and loss_t > .005):
                

                mean,chole, probs = self.policy.forward(states_old)
                pi = Categorical(probs=probs)
                dist_c = MultivariateNormal(mean, scale_tril = chole)

                logprobs_d = pi.log_prob(actions_d)
                logprobs_c = dist_c.log_prob(actions_c).squeeze()
                logprobs = logprobs_d + logprobs_c
                  
                loss_p = torch.mean(
                    qij * logprobs)
                    
                # Forward KL Divergence (target | present)
                kl = self._discrete_kl(p1=probs_t, p2=probs)    # FIXED
                kl_μ, kl_Σ = self.gaussian_kl(μi=mean_t, μ=mean, \
                                         Ai=chole_t, A=chole)

               # Update lagrange multipliers by gradient descent
                self.eta_kl -= self.alpha_d * (self.epsilon_kl - kl).detach().item()
                self.η_kl_μ -= self.alpha_c_μ * (self.ε_kl_μ - kl_μ).detach().item()
                self.η_kl_Σ -= self.alpha_c_Σ * (self.ε_kl_Σ - kl_Σ).detach().item()

                if self.η_kl_μ < 0.0:
                    self.η_kl_μ = 0.0
                if self.η_kl_Σ < 0.0:
                    self.η_kl_Σ = 0.0
                if self.eta_kl < 0.0:
                    self.eta_kl = 0.0
                                             
                self.loss_tracker.append(loss_p.detach().cpu().numpy())

                if share_local:                
                    # Forward KL discrete (target | present)
                    kl_group = self._discrete_kl(p1=group_probs, p2=probs)   
                    kl_µ_group, kl_S_group = self.gaussian_kl(µi=group_mean, µ=mean, \
                                             Ai=group_chole, A=chole)
                                                                    
                    # Limit to positive or 0 values
                    kl_group = torch.max(kl_group, zero_tensor)
                    kl_µ_group = torch.max(kl_µ_group, zero_tensor)
                    group_kl = ( self.zeta_disc* kl_group +  self.zeta_cont * kl_µ_group)
                    
                else:
                    group_kl = kl_group = kl_µ_group = 0

                self.policy_optimizer.zero_grad()
                loss_policy = -(loss_p + self.eta_kl * (self.epsilon_kl  - kl) \
                                + self.η_kl_μ * (self.ε_kl_μ - kl_μ) \
                                + self.η_kl_Σ * (self.ε_kl_Σ - kl_Σ)
                                - group_kl)
               
                loss_policy.backward()
                clip_grad_norm_(self.policy.parameters(), self.grad_norm)
                self.policy_optimizer.step()
                mean_policy += loss_policy.item()
                mean_loss_p += -loss_p.item()
                actorLearn += 1
                  
                loss_t = abs((self.loss_tracker[-2] - self.loss_tracker[-1])/self.loss_tracker[-1])
                li+=1
        
        # Update critic once per batch with lower tau (small steps)
        self.upate_critic_target_network(self.tau_critic)

      self.update_actor_target_network(self.tau_actor) # Update the actor with higher tau
      print(f'Q Loss {mean_q_loss/criticLearn}, Loss P {loss_p:.4f}   eta D {self.eta_kl:.2f} eta C {self.η_kl_μ:.2f}  eta sig {self.η_kl_Σ:.2f} ')
      print('*')
      return mean_q_loss/criticLearn, mean_policy/actorLearn, mean_loss_p/actorLearn, self.η_kl_μ, self.η_kl_Σ, self.eta_kl, group_kl, self.zeta_disc* kl_group, self.zeta_cont * kl_µ_group
    

    def update_target_networks(self,tau = 1):
      # update target networks 
      for target_param, param in zip(self.policy_target.parameters(), self.policy.parameters()):
          target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            
      for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
          target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
          
    def update_actor_target_network(self,tau = 1):
      for target_param, param in zip(self.policy_target.parameters(), self.policy.parameters()):
          target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def upate_critic_target_network(self,tau = 1):
      for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
          target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def run_training(self, num_steps_in_train_episode, data_num, episode):

        train_cycle_num = 4  # Set this for the cycle type (4 - suburban)
        
        data = np.empty((int(num_steps_in_train_episode), 32)) 
        data[:] = np.nan
        state = self.env.randomize_drive_cycle(train_cycle_num, 25)

        done = is_trunc = timeout = False
        step_track = episode_steps = 0 

        episode_reward = 0
        eps_rewards_accel = 0
        eps_rewards_fuel = 0
        eps_rewards_torque = 0
        eps_rewards_gear = 0
        eps_rewards_gearpenalty = 0
        eps_rewards_gearFlutter = 0
        
        timeout = False
        while not timeout:
            with torch.no_grad():
              action, d_probs, logprob_c, logprob_d = self.policy_target.select_action(state) 
              
            action = action.squeeze().cpu().detach().numpy() # (n_envs, nd + nc)
            action_env = np.zeros_like(action)      

            action_env[0] = np.clip(action[0] * self.torque_high, self.torque_low, self.torque_high)
            action_env[1] = action[1] - 1 # hard coding for gear choice -> [-1 0 1]
            
            # States: Vobj - Pego, Vego, Aego, Ades, gear, Vobj, Ttrac_Actual 
            new_state, reward, is_trunc, info = self.env.step(action_env)
            
            timeout = (episode_steps+1) >= num_steps_in_train_episode
            
            experience =  (state.astype(np.float16), action.astype(np.float16), reward, new_state.astype(np.float16), False, logprob_c.detach().cpu().numpy().astype(np.float32), logprob_d.detach().cpu().numpy().astype(np.float32), timeout)

            self.memory.store_trajectory(experience)
            
            episode_reward += reward  
            eps_rewards_accel += info['R1'] 
            eps_rewards_fuel += info['R3'] 
            eps_rewards_torque += info['R2'] 
            eps_rewards_gear += info['Rg'] 
            eps_rewards_gearpenalty += info['R5']
            eps_rewards_gearFlutter += info['R6']
                        

            train_fuel_consumption = info['fuel_consumption']
            dV = info['obj_velocity'] - info['velocity']
            # Step data to save
            data[episode_steps,:] = info['time'], info['dX'], dV, info['velocity'],info['acceleration'], info['position'], info['energy'],\
                info['obj_velocity'], info['obj_position'], info['t_traction'], info['gear'], action_env[0],\
                info['engine_torque'], info['des_accel'], info['brake_torque'], info['eng_spd'], action_env[1],  info['w_in_gear'], info['g_control'],\
                reward, info['R1'], info['R2'], info['R3'], info['R4'], info['Rg'], info['mass'], info['crash'], info['drive_cycle'], info['R5'], info['R6'], info['cycle_time'],0


            state = new_state # don't forget to update the state!

            episode_steps += 1
            step_track += 1 # to figure out when to call learn
            self.total_steps += 1 # to figure out when to call learn
            
            if step_track > self.sample_steps:
              
                mean_q_loss, mean_policy, mean_loss_p, eta_kl_c, eta_kl_sigma, eta_kl_d, loss_p_r, kl_f, kl_μ_f  = self.learn(data_num)
                
                step_track = 0
                self.mean_q_loss.append(mean_q_loss)
                self.mean_policy.append(mean_policy)
                self.mean_loss_p.append(mean_loss_p)
                self.final_eta_kl_c.append(eta_kl_c)
                self.final_eta_kl_d.append(eta_kl_d)
                self.final_eta_kl_sig.append(eta_kl_sigma)
                self.episode_tracker.append(episode)
                self.final_loss_p_r.append(loss_p_r)
                self.final_kl_d_f.append(kl_f)
                self.final_kl_c_f.append(kl_μ_f)
                
                step_track = 0
                
                loss_tracking =  {
                      'mean_q_loss': self.mean_q_loss,
                      'mean_policy': self.mean_policy,
                      'mean_loss_p': self.mean_loss_p,
                      'eta_kl_c': self.final_eta_kl_c,
                      'eta_kl_sig':self.final_eta_kl_sig,
                      'eta_kl_d': self.final_eta_kl_d,
                      'episode_tracker': self.episode_tracker, 
                      'f_kl_d' : self.final_kl_d_f, 
                      'f_kl_c' : self.final_kl_c_f,
                      'loss_p_r' : self.final_loss_p_r, 
                      'total_steps' : self.total_steps}
                      
                self.save_current_policy(data_num)
                #print(f'************** SAVED DATA : {data_num}  AGENT NUM {self.number}   EPISODE {episode} LEARN STEPS {self.learn_steps_taken}  Device {self.device}  *****************')
                #print(f'Episode {episode} Agent {self.number}  Q loss : {mean_q_loss:.3f}, policy loss : {mean_policy:.3f} KL Loss {loss_p_r:.3f} rKLd {kl_f:.3e}  rKLmu {kl_μ_f:.3e}')

        # Save training episode data
        mean_reward_per_step_train = episode_reward / episode_steps
        self.eps_train_results = self.eps_train_results.copy()
        self.eps_train_results[episode-1,:] =  episode, episode_reward, episode_steps, info['mpg'],\
            eps_rewards_accel, eps_rewards_fuel, eps_rewards_torque,\
            eps_rewards_gear, eps_rewards_gearpenalty, eps_rewards_gearFlutter, info['position'],\
            train_cycle_num, train_fuel_consumption, mean_reward_per_step_train, info['mass'], info['drive_cycle']
        
        # Evaluate Current Policy    
        eval_reward, eval_mpgs, eval_episode_steps, eval_final_pos, eval_crash, a_error, g_count, g_flutter,\
            shift_per_m, rewards_step, eval_data, eval_rewards_accel, eval_rewards_fuel, eval_rewards_torque, eval_rewards_gear, eval_rewards_gearpenalty,\
            eval_rewards_gearFlutter = self.evaluate(stoch = False, cycle_num = 4)
        
        # Save evaluation episode data  
        self.eps_eval_results = self.eps_eval_results.copy()             
        self.eps_eval_results[episode-1, :] = episode, eval_reward, eval_final_pos, eval_mpgs,\
                        eval_rewards_accel, eval_rewards_fuel, eval_rewards_torque, eval_rewards_gear,\
                        eval_rewards_gearpenalty, eval_rewards_gearFlutter,rewards_step,\
                        a_error, g_count, g_flutter, shift_per_m, rewards_step
        
        # Save Data File
        if (episode % 2) == 0:
            #currentDir = os.getcwd()
            dirName = r'data/gen_{}'.format(data_num)
            path = os.path.join(CURR_DIR, dirName)
            if not os.path.exists(path):
                os.mkdir(path)
            filename = r'AMT_Central_Shared_Suburban_{}_Agent{}_eps{}.mat'.format(data_num, self.number, episode)
            save_path  = r"{}data/gen_{}/{}".format(CURR_DIR,data_num, filename) 
                
            sio.savemat(save_path, {'info': self.run_info, 'eps_eval' : self.eps_eval_results[:episode], 'eps_train':self.eps_train_results[:episode], 'data': data, 'evaluation_data' : eval_data,  'loss_tracking' : loss_tracking, 'fleet_num' : data_num})
                           
                  
        # Save model every 10 episodes    
        if (episode % 10) == 0:
                # Save the model parameters 
                self.save_model(episode, data_num)
        return eval_mpgs

    # Evaluate greedy policy
    def evaluate(self, roll_outs = 1, stoch = False, cycle_num = 4):
      
      g_count = 0
      g_flutter = 0
      p_gear = 1
      eval_crash = 0
      gear_switching = np.zeros([3])

      estate = self.env.evaluation_drive_cycle(cycle_num)     
      
      max_eval_steps = self.env.max_episode_steps
      eval_data = np.empty((max_eval_steps, 40)) 
      eval_data[:] = np.nan
      
     # Could add multiple runs
      for i in range(1):
          episode_steps = 0

          # Track components of rewards
          eval_reward = 0 # keep track of the rollout reward 
          eval_rewards_accel = 0 
          eval_rewards_fuel = 0 
          eval_rewards_torque = 0  
          eval_rewards_gear = 0
          eval_rewards_gearpenalty = 0
          eval_rewards_gearFlutter = 0
          eval_crash = 0
                    
          timeout = False          
          while not timeout:

              with torch.no_grad():
                action, d_probs = self.policy.select_greedy_action(estate)
              action = action.squeeze().cpu().detach().numpy() # (n_envs, nd + nc)
              b_vals = d_probs.squeeze().cpu().detach().numpy() # (n_envs, nd + nc)

              action_env = np.zeros_like(action)                    
              action_env[0] = np.clip(action[0] * self.torque_high, self.torque_low, self.torque_high)
              action_env[1] = action[1] - 1 # hard coding for gear choice -> [-1 0 1]

              estate, reward, timeout, info = self.env.step(action_env)
          
              episode_steps +=1 

              eval_reward += reward 
              eval_rewards_accel += info['R1'] 
              eval_rewards_fuel += info['R3'] 
              eval_rewards_torque += info['R2'] 
              eval_rewards_gear += info['Rg'] 
              eval_rewards_gearpenalty += info['R5']
              eval_rewards_gearFlutter += info['R6']
              eval_crash += info['crash']

     
              # Save data to evaluation file
              dV = info['obj_velocity'] - info['velocity']
              eval_data[episode_steps-1,:] = info['time'], info['dX'], dV, info['velocity'],info['acceleration'], info['position'], info['energy'],\
                info['obj_velocity'], info['obj_position'], info['t_traction'], info['gear'], action_env[0],\
                info['engine_torque'], info['des_accel'], info['brake_torque'], info['eng_spd'], action_env[1],  info['w_in_gear'], info['g_control'],\
                reward, info['R1'], info['R2'], info['R3'], info['R4'], info['Rg'], info['mass'], info['crash'], 1, info['R5'], info['R6'],\
                info['cycle_time'], info['eval_drive_cycle'], b_vals[0], b_vals[1], b_vals[2], 0, 0, 0, 0, 0
                  
              # check for fluttering if contains a 1 and a -1 within last 3 steps  (t-2, t-1, t)
              gear_switching[:] = gear_switching[1], gear_switching[2], info['gear'] - p_gear   # gear attained - prev gear
         
              if np.array_equal(np.abs(gear_switching),np.array([0, 1 ,1])) or np.array_equal(np.abs(gear_switching),np.array([1, 1 ,1])):
                  g_flutter += 1 
                  
              g_count += abs(info['gear'] - p_gear)
              p_gear = info['gear']

              if info['terminate']:
                timeout = True   
                
          mpgs = info['mpg']
          eval_final_pos = info['position']
          d_accel = np.subtract(eval_data[:episode_steps - 1,4], eval_data[:episode_steps - 1,13])
          a_rsme = np.sqrt(np.square(d_accel).mean()) 
          shift_per_m = g_count / max(1, eval_final_pos )  
          rewards_step =  eval_reward/ episode_steps
          eval_data = eval_data[:episode_steps-1,:]
         
      return eval_reward, mpgs, episode_steps, eval_final_pos, eval_crash, a_rsme, g_count, g_flutter, shift_per_m, rewards_step, eval_data,\
            eval_rewards_accel, eval_rewards_fuel, eval_rewards_torque, eval_rewards_gear, eval_rewards_gearpenalty, eval_rewards_gearFlutter

    def save_model(self, episode_id, data_num):
        data = {
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.critic_target.state_dict(),
            'actor_state_dict': self.policy.state_dict(),
            'target_actor_state_dict': self.policy_target.state_dict(),
            'critic_optim_state_dict': self.critic_optimizer.state_dict(),
            'actor_optim_state_dict': self.policy_optimizer.state_dict(),
            'memory': self.memory,
            'eps_train_results': self.eps_train_results, 
            'eps_eval_results' : self.eps_eval_results
 
        }
        filename = 'AMT_Central_RL_Agent{}_{}.{}.pt'.format(self.number, data_num, episode_id)
        currentDir = os.getcwd()
        dirName = '{}models/gen_{}'.format(CURR_DIR, data_num)
        path = os.path.join(currentDir, dirName)
        if not os.path.exists(path):
            os.mkdir(path)
        self.save_path  = F"{CURR_DIR}models/gen_{data_num}/{filename}" 
        torch.save(data, self.save_path)
        
    def save_current_policy(self, data_num):
          data = {
              'critic_state_dict': self.critic.state_dict(),
              'actor_state_dict': self.policy.state_dict(),
              'saved_state_dist': self.saved_state_dist   
          }
          filename = 'AMT_Shared_Policy_{}.{}.pt'.format(data_num, self.number)
          currentDir = os.getcwd()
          dirName = '{}/agent_models/gen_{}'.format(CURR_DIR, data_num)
          path = os.path.join(currentDir, dirName)
          if not os.path.exists(path):
              os.mkdir(path)
          save_path  = F"{dirName}/{filename}" 
          torch.save(data, save_path)

    def load_model(self, filename, data_num): 

        
        self.save_path  = F"models/gen_{data_num}/{filename}"  
        checkpoint = torch.load(self.save_path, map_location=torch.device(self.device))
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['target_critic_state_dict'])
        self.policy.load_state_dict(checkpoint['actor_state_dict'])
        self.policy_target.load_state_dict(checkpoint['target_actor_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optim_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['actor_optim_state_dict'])
        self.memory = checkpoint['memory']

        
    def load_model_policy(self, filename): 
        
        self.save_path  = "{}group_models/pre_agent/{}".format(CURR_DIR, filename)  
        checkpoint = torch.load(self.save_path, map_location=torch.device(self.device))
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['target_critic_state_dict'])
        self.policy.load_state_dict(checkpoint['actor_state_dict'])
        self.policy_target.load_state_dict(checkpoint['target_actor_state_dict'])


    def load_group_policy(self, data_num): 
        
        filename = 'AMT_GroupModel_{}.pt'.format(data_num)
        save_path  = r"{}/group_models/gen_{}/{}".format(CURR_DIR, data_num, filename)
        
        if not os.path.isfile(save_path):
          print('No group Policy Found, use older version')
          return self.group_policy, False

        for n in range(10):
            print(r'Attempt # {n}')
            try:
                checkpoint = torch.load(save_path, map_location=torch.device(self.device))

            except:
                print('Retry Loading Agent')
                time.sleep(n)
            else: 
                print(f'Loaded new group policy for agent {self.number}')
                self.group_policy.load_state_dict(checkpoint['actor_state_dict'])
                return self.group_policy, True
        return self.group_policy, False
               