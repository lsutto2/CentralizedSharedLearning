# -*- coding: utf-8 -*-
"""
Auxilliary Classses

Memory Replay Buffer

@author: Kerbel 2021
"""
import numpy as np

class Memory():
    def __init__(self, max_size, input_shape, n_actions,n_envs=1):
    
        # Allows for vectorized environments (not currently using)
        self.n_envs = n_envs
        # mem_size is total number of experiences held. so divide max_size by n_envs
        self.mem_size = int(np.ceil(max_size)) 
        self.max_size = self.mem_size
        self.n_states = input_shape  
        self.n_actions = n_actions
        
        self.clear_memory()
        
    def store_trajectory(self, experience):
        # Modules of current counter and size -> Replace older
        state, action, reward, state_, done, logprob_c, logprob_d, trunc = experience
        
        if state.ndim == 1:
              state = np.expand_dims(state, axis = 0)
              state_ = np.expand_dims(state_, axis = 0)
              action = np.expand_dims(action, axis = 0)
        
        
        #state, action, reward, state_, done, logprob, trunc = experience
        state = np.transpose(state) # (nS,n_envs)
        action = np.transpose(action) # (nc + nc,n_envs)
        state_ = np.transpose(state_) # (nS,n_envs)

        self.state_memory[self.next_idx] = state
        self.new_state_memory[self.next_idx] = state_
        self.action_memory[self.next_idx] = action
        self.reward_memory[self.next_idx] = reward
        self.done_memory[self.next_idx] = done
        self.logprob_c_memory[self.next_idx] = logprob_c
        self.logprob_d_memory[self.next_idx] = logprob_d
        self.truncated_memory[self.next_idx] = trunc
      
        self.next_idx = (self.next_idx + 1) % self.mem_size
        self.num_in_buffer = min(self.max_size, self.num_in_buffer + self.n_envs)
        
    def sample_trajectory(self, last_steps):
        sequence_steps = 1
        
        last_steps = last_steps + 1
        if self.next_idx - last_steps - 1 <= 0:
          first_idxs = np.arange(0, self.next_idx)
          newidxs = last_steps - self.next_idx
          last_idxs = np.arange(self.mem_size - newidxs, self.mem_size)
          idxs = np.hstack([last_idxs, first_idxs])
        else:
          idxs = np.arange(self.next_idx - last_steps - 2, self.next_idx - 2, 1)  # start, end, step 
        # make the 1-D indices into 2-D (index, specfic_env_num)
        idxs,env_num = np.unravel_index(idxs, self.truncated_memory.shape)     
        
        states = np.zeros((last_steps, sequence_steps, self.n_states))
        states_ = np.zeros((last_steps, sequence_steps, self.n_states))
        actions = np.zeros((last_steps, sequence_steps,self.n_actions))
        rewards = np.zeros((last_steps, sequence_steps))
        logProbs_d = np.zeros((last_steps, sequence_steps))
        logProbs_c = np.zeros((last_steps, sequence_steps))
        dones = np.zeros((last_steps, sequence_steps))
        is_truncated = np.zeros((last_steps, sequence_steps))
  
        for i, idx in enumerate(idxs): 
          states[i] = self.state_memory[idx:idx + sequence_steps,:,env_num[i]]
          states_[i] = self.new_state_memory[idx:idx + sequence_steps,:,env_num[i]]
          actions[i] = self.action_memory[idx:idx + sequence_steps,:,env_num[i]]
          rewards[i] = self.reward_memory[idx:idx + sequence_steps,env_num[i]]
          logProbs_d[i] = self.logprob_d_memory[idx:idx + sequence_steps,env_num[i]]
          logProbs_c[i] = self.logprob_c_memory[idx:idx + sequence_steps,env_num[i]]
          dones[i] = self.done_memory[idx:idx + sequence_steps,env_num[i]]
          is_truncated[i] = self.truncated_memory[idx:idx + sequence_steps,env_num[i]]
        
        return states, actions, rewards, states_, dones, logProbs_c, logProbs_d, is_truncated

        
    def sample_random_steps(self, batch_size, sequence_steps):
        # Uniformly sample a batch from the memory
        idxs1D = np.random.randint(0, self.num_in_buffer-sequence_steps, batch_size+2*sequence_steps)
        
        # make the 1-D indices into 2-D (index, specfic_env_num)
        idxs,env_num = np.unravel_index(idxs1D, self.truncated_memory.shape)
        
        OOB = ((idxs <= self.next_idx) & (idxs >= self.next_idx-sequence_steps)) |  (idxs >= self.num_in_buffer/self.n_envs-sequence_steps)
        inBounds = np.squeeze(np.argwhere(~OOB))
        idxs = idxs[inBounds[:batch_size]]
        env_num = env_num[inBounds[:batch_size]]
        
        if not idxs.size==batch_size:
            print(f'ERROR:  Not the right size  {idxs.size}')
        
        states = np.zeros((batch_size, sequence_steps, self.n_states))
        states_ = np.zeros((batch_size, sequence_steps, self.n_states))
        actions = np.zeros((batch_size, sequence_steps,self.n_actions))
        rewards = np.zeros((batch_size, sequence_steps))
        logProbs_d = np.zeros((batch_size, sequence_steps))
        logProbs_c = np.zeros((batch_size, sequence_steps))
        dones = np.zeros((batch_size, sequence_steps))
        is_truncated = np.zeros((batch_size, sequence_steps))

  
        for i, idx in enumerate(idxs): 
          states[i] = self.state_memory[idx:idx + sequence_steps,:,env_num[i]]
          states_[i] = self.new_state_memory[idx:idx + sequence_steps,:,env_num[i]]
          actions[i] = self.action_memory[idx:idx + sequence_steps,:,env_num[i]]
          rewards[i] = self.reward_memory[idx:idx + sequence_steps,env_num[i]]
          logProbs_d[i] = self.logprob_d_memory[idx:idx + sequence_steps,env_num[i]]
          logProbs_c[i] = self.logprob_c_memory[idx:idx + sequence_steps,env_num[i]]
          dones[i] = self.done_memory[idx:idx + sequence_steps,env_num[i]]
          is_truncated[i] = self.truncated_memory[idx:idx + sequence_steps,env_num[i]]
        
        return states, actions, rewards, states_, dones, logProbs_c, logProbs_d, is_truncated

    # Clear memory
    def clear_memory(self):      
        # Keep track of first available space to store the memory
        self.mem_cnt = 0 # assume synchronous sampling of environments

        self.state_memory = np.zeros((self.mem_size, self.n_states, self.n_envs)) # (mS, oS, nE)
        self.new_state_memory = np.zeros((self.mem_size, self.n_states, self.n_envs)) # (mS, oS, nE)
        # Not discrete, number of components to the action 
        self.action_memory = np.zeros((self.mem_size, self.n_actions, self.n_envs)) # (mS, nA, nE)
        self.reward_memory= np.zeros((self.mem_size, self.n_envs)) # (mS, nE)
        self.done_memory = np.zeros((self.mem_size, self.n_envs)) # (mS, nE)
        self.logprob_c_memory = np.zeros((self.mem_size, self.n_envs)) # (mS, nE)
        self.logprob_d_memory = np.zeros((self.mem_size, self.n_envs)) # (mS, nE
        self.truncated_memory = np.zeros((self.mem_size, self.n_envs)) # (mS, nE)
        # Size indexes
        self.next_idx = 0
        self.num_in_buffer = 0
        
