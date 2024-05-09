# -*- coding: utf-8 -*-
"""
DeterministicTrack Class
- Open AI Gym custom environment
Created on Wed Oct 27 13:17:10 2021

@author: linds
"""

# Environment Setup
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from scipy import interpolate
import scipy.io as sio
from VehicleModelClass import Vehicle_Model


DIESEL_D = 0.835
CURR_DIR = '/scratch/lsutto2/AMT_Central_SubSize/'

# Using a Custom Gym Environment
class DeterministicTrack(gym.Env):


    def __init__(self, reward_weights, rank = 0, seed = 0, cycle_num = 4, car_id=1, useAlt = 1, ignoreDone = True):
        """
        The car is longitudinal guided with uniformly accelerated motion over the time step. The car is setup to follow a leading
        car using the Intelligent Driver Model (IDM) where the leading vehicle follows the given drive cycle.
        """
        self.ego_vehicle = Vehicle_Model(car_id=car_id)
        self.car_id = car_id
        self.ignoreDone = ignoreDone  # Repeat the drive cycle without stopping

        self.reward_weights = reward_weights
        # Set random variables
        self.rng = np.random.default_rng(seed + rank)
        self.seed(seed + rank)
        self.rank = rank # (0- # of environments)
         

        self.dt = .5 # Delta Time step
        self.sensor_range = 150.0  # meters to see leading car (IDM)


        # Intitial conditions
        self.position = 1.0  # in m; track is 1-dim, only one coord is needed
        self.velocity = 0.0  # in m/s
        self.acceleration = 0.0  # in m/s**2
        self.gear = 1
        self.prev_acceleration = 0.0  # in m/s**2
        self.time = 0  # in s
        self.fuel_rate = 0
        self.Wengine = 650 # in RPM
        self.total_fuel_g = 0
        self.obj_position = 100
        self.obj_velocity = 0
        self.ego_desired_velocity = 35  # Nax speed when not following a vehicle
        self.ego_desired_acceleration = 0
        self.mpg = 0
        self.fuel_consumption = 0
        self.Tengine = 0
        self.prev_T_traction = 0
        self.actual_traction_torque = 0
        self.prev_ctorque = 0
        self.ego_desired_acceleration_previous = 0
        self.gear_previous = 1
        self.gear_prev = np.array([1, 1, 1])
        self.ugear_prev = np.array([0, 0, 0])
        self.i = self.i_eval = cycle_num   # Route to train

        self.stoch = False
        self.velocity_noise = self.rng.random() * 3
        self.dV = 0
        self.dL = self.obj_position - self.position
        
        # Actions: traction_torque, delta gear (-1, 0, +1)
        self.action_space = spaces.Box(np.array([-30000,-1]), np.array([30000, 1]))

        # Driver Only action space (uses IDM to torque and fuel minimized gear)
        #self.action_space = spaces.Box(np.array([-50000, 1]), np.array([50000, 10]))


        # States: Vego, Aego, Ades, gear, Aego_prev, gear_prev, Mass
        self.observation_low = np.array([0, -3, -3, 1, -3, 1, 5000])
        self.observation_high = np.array([38, 3, 3, 10, 3, 10, 25000])
        self.observation_space = spaces.Box(self.observation_low, self.observation_high)
 
        self.viewer = None
        self.useAlt = useAlt
        self.train_time = 0
      
        
    # Randomly sample a drive cycle from the corresponding folder
    def randomize_drive_cycle(self, cycle, num):
        self.stoch = True
        self.ignoreDone = True
        i = self.rng.integers(1, num, 1)
        self.i = i[0]
        filename = '{}drive_cycles_{}/drive_cycle_{}_num_{}.mat'.format(CURR_DIR, cycle, cycle, self.i)
        route_path  = F"{filename}" 
 
        drive_cycle = sio.loadmat(route_path)
        ref_velocity = np.array(drive_cycle['v_cyc']).squeeze()
        cycle_time = np.array(drive_cycle['t_cyc']).squeeze()
        cycle_time_plus = np.array(list(range(cycle_time[-1] + 1, cycle_time[-1] + 201)))
        cycle_time_new = np.append(cycle_time, cycle_time_plus)
        self.max_cycle_time = np.max(cycle_time_new)
        ref_velocity_plus = ref_velocity[0:200]
        ref_velocity_new = np.append(ref_velocity, ref_velocity_plus)
        self.drive_cycle_f = interpolate.interp1d(cycle_time_new, ref_velocity_new)
        self.cycle_position_end = np.sum(ref_velocity) - 100
        
        self.max_episode_steps = round(self.max_cycle_time / self.dt)
        
        return self.reset()
    
    # Set deterministic drive cycle used for evaluations        
    def evaluation_drive_cycle(self, cycle_num = 4):
    
        self.ego_vehicle = Vehicle_Model()
        self.ego_vehicle.mass = 13000 # reset vehicle mass
        self.stoch = False  # Deterministic
        self.ignoreDone = False  # Stop when drive cycle ends

        filename = 'drive_cycle_{}.mat'.format(cycle_num)
        route_path  = F"{CURR_DIR}DriveCycles/MyCycles/{filename}"
        drive_cycle = sio.loadmat(route_path)
        ref_velocity = np.array(drive_cycle['v_cyc']).squeeze()
        cycle_time = np.array(drive_cycle['t_cyc']).squeeze()
        cycle_time_plus = np.array(list(range(cycle_time[-1] + 1, cycle_time[-1] + 201)))
        cycle_time_new = np.append(cycle_time, cycle_time_plus)
        self.max_cycle_time = np.max(cycle_time_new)
        ref_velocity_plus = ref_velocity[0:200]
        ref_velocity_new = np.append(ref_velocity, ref_velocity_plus)
        self.drive_cycle_f = interpolate.interp1d(cycle_time_new, ref_velocity_new)
        self.max_distance = np.max(np.array(drive_cycle['x_cyc']).squeeze())
        self.max_episode_steps = round(self.max_cycle_time / self.dt)

        return self.reset()

    # One step in the environemnt
    def step(self, action):

        # Get desired actions
        T_traction = action[0]
        c_torque = action[0]  # to penalize in reward, not changed 
        gear_delta = int(action[1])
        reward = 0

        # Keep track of previous gear
        self.gear_previous = self.gear

        # Limit gear to possible range
        g_control = self.gear + gear_delta
        self.gear =  np.clip(self.gear + gear_delta, 1, 10)

        # Get engine speed in selected gear before it changes
        w_in_gear = self.ego_vehicle.get_engineSpeed_in_gear(self.velocity, self.gear)

        if T_traction > 0:
          # Get the engine torque (using previous velocity for Wengine to find maximum)
          self.Tengine, self.gear = self.ego_vehicle.get_engine_torque(T_traction, self.velocity, self.gear)
          self.Tbrake = 0
        
        else:            
          self.Tbrake, self.Tengine, self.gear = self.ego_vehicle.get_actual_brake_torque(T_traction, self.velocity, self.gear) 

        # Calculate fuel rate
        self.fuel_rate = self.ego_vehicle.get_fuelrate(self.velocity, self.Tengine, self.gear, self.dt)
        self.fuel_consumption += self.fuel_rate*self.dt
        self.mpg = (self.position/max(0.0001,self.fuel_consumption/DIESEL_D*1e-6))/(1609*264)

        # Calculate remaining torque from engine to the wheel (accessories, etc)
        tengine_to_wheel = self.ego_vehicle.get_remaining_torque_toWheel(self.Tengine, self.velocity, self.gear, T_traction)

        self.actual_traction_torque = tengine_to_wheel + self.Tbrake   # Only used for data collection

        # Calculate acceleration from torque to wheel 
        self.acceleration = self.ego_vehicle.calc_acceleration_from_tt(self.velocity, tengine_to_wheel, self.Tbrake, self.gear)
        # s = 0.5 * a * tÂ² + v0 * t + s0
        self.position += (0.5 * self.acceleration * self.dt**2 + self.velocity * self.dt)
        self.position = np.maximum(0, self.position)
        
        # v = a * t + v0
        self.velocity += self.acceleration * self.dt
        self.velocity = np.maximum(0, self.velocity)  # Never go backwards

        self.time += self.dt
        self.train_time += self.dt
        self.dX = self.obj_position - self.position   #Only used for data

        # Setup Reward function
        # Previous desired acceleration (S) - new acceleration (S')
        d_accel = self.ego_desired_acceleration - self.acceleration
        self.gear_prev = np.array([self.gear_prev[1], self.gear_prev[2], self.gear])  # [t-2, t-1, t] 
        self.ugear_prev = np.array([self.ugear_prev[1], self.ugear_prev[2], gear_delta])
        
        # Penalty weights: Acceleration Error, Torque Control error, fuel rate, 
        # shifting frequency, power reserce, Wrong gear, gear fluttering 
        Wa, Wtt, gamma, Wg, Wp, Wgp, Wfl = self.reward_weights

        # First we normalize the reward parameters and use absolute error 
        max_d_accel = 4
        r_accel = abs(d_accel)/max_d_accel
        if self.velocity < .001 and self.ego_desired_acceleration < 0:
          r_accel = 0
        
        max_abs_torque = 20000
        r_torque = abs(c_torque)/max_abs_torque
       
        max_fuel = 18
        r_fuel = self.fuel_rate / max_fuel
        
        # minimize difference from max power reserve
        self.eng_spd, self.gear = self.ego_vehicle.get_engineSpeed(self.velocity, self.gear)
        Tmax = self.ego_vehicle.get_maxTengine(self.eng_spd)
        max_pr = 334
        pr_des = -.25 * self.velocity**2 + 2.53*self.velocity + 327.2
        r_power = (Tmax - self.Tengine) * self.eng_spd * np.pi/30 / 1000  # kW   
        r_power = abs(pr_des - r_power) / max_pr 

        R1 = Wa *r_accel
        R2 = Wtt *r_torque
        R3 = gamma *r_fuel
        R4 = Wp *r_power
        R5 = 0
        Rg = 0
        R6 = 0
        
        # Penalize shifting frequency over 3 steps
        r_gear = max(np.sum(np.abs(self.ugear_prev)) - 1, 0)   # Correct  (000 - 1, max 0), (111 - 1, max 2) 
        Rg = Wg * r_gear

        if self.gear != g_control:
          R5 = Wgp * 1
          
        r_flutter = 0        
        if self.ugear_prev[2] != 0:
            if np.array_equal(self.ugear_prev,np.array([-1, 1 ,-1])) or np.array_equal(self.ugear_prev,np.array([1, -1 ,1])):
                  r_flutter = 1
                  
            if np.array_equal(self.ugear_prev[1:2],np.array([-1, 1])) or np.array_equal(self.ugear_prev[1:2],np.array([1, -1])):
                  r_flutter = 1
        R6 = Wfl * r_flutter

        # There is no penalty for car following, but could be added here
        reward = - (R1 + R2 + R3 + R4 + Rg + R5 + R6)
        
        # Get the desired acceleration from IDM for next step
        self.ego_desired_acceleration_previous = self.ego_desired_acceleration    # Use this as a state?
        self.ego_desired_acceleration = self.driver_model_IDM()
        self.eng_spd, _ = self.ego_vehicle.get_engineSpeed(self.velocity, self.gear)
        self.prev_ctorque = c_torque  
        terminate = False
        
        if self.obj_position <= self.position:
          crash = 1 # Only recording crashes currently - could add as penalty
        else: crash = 0
        
        state = self.get_state()
        # Used to save data
        info = {
            'position': self.position,
            'dX' : self.dX,
            'velocity': self.velocity,
            'acceleration': self.acceleration,
            'des_accel': self.ego_desired_acceleration,
            'cycle_time': self.time,
            'time': self.train_time,
            'energy': self.fuel_rate,
            'obj_position' : self.obj_position,
            'obj_velocity' : self.obj_velocity,
            'fuel_consumption' : self.fuel_consumption,
            'mpg' : self.mpg,
            'engine_torque' : self.Tengine,
            'brake_torque' : self.Tbrake, 
            'eng_spd' : self.eng_spd, 
            'gear' : self.gear,
            't_traction' : self.actual_traction_torque, 
            'R1' : R1,
            'R2' : R2,
            'R3' : R3,
            'R4' : R4, 
            'Rg' : Rg, 
            'R5' : R5,
            'R6' : R6,
            'ref_vel' : self.desired_velocity, 
            'w_in_gear' : w_in_gear,
            'g_control' : g_control,
            'mass' : self.ego_vehicle.mass, 
            'crash' : crash,
            'terminate' : terminate, 
            'drive_cycle' : self.i, 
            'eval_drive_cycle' : self.i_eval
            
        }
        
        done = False
        if self.ignoreDone:
            if self.time > self.max_cycle_time - 1:
              info['terminate'] = True
              # Reset the drive cycle
              #print('Reset driver cycle')
              self.time = 0
              self.reset_mass(self.stoch)
              #state = self.get_state()
              return state, reward, done, info # True is truncated - starting over
        else: # Position only matters in evaluation
            if self.position >  self.max_distance:
              done = True
              print(f' pos: {self.position}')
            if self.time > self.max_cycle_time - 1:
              done = True
              print(f' time: {self.time}')
        

        return state, reward, done, info

    def reset(self, seed = None):
    
        
        self.position = 5.0
        self.velocity = 0.0
        self.acceleration = 0.0
        self.gear = 1
        self.prev_acceleration = 0.0
        self.ego_desired_acceleration_previous = 0
        self.fuel_rate = 0.0  # in Wh
        self.Wengine = 650 # in RPM
        self.total_fuel_g = 0
        self.time = 0
        self.obj_position = 100
        self.obj_velocity = 0
        self.ego_desired_acceleration = 0
        self.Tengine = 0
        self.Tbrake = 0
        self.actual_traction_torque = 0
        self.fuel_consumption = 0 
        self.mpg = 0
        self.prev_ctorque = 0
        self.ego_vehicle = Vehicle_Model(car_id=self.car_id)
        self.gear_previous = 1
        self.train_time = 0
        self.reset_mass(self.stoch)
        self.dX = self.obj_position - self.position
        state = self.get_state()
        return state
    
    # Mass changes every route with stochastic setting
    def reset_mass(self, stoch):
        self.ego_vehicle.set_base_mass(self.car_id)
        if stoch: 
            self.ego_vehicle.set_base_mass(2)  # Start from middle 16k
            self.mass_noise = self.rng.integers(-8000, 8000, 1)
            self.ego_vehicle.mass += self.mass_noise[0]


    def get_state(self):
        """
        Wrapper to update state with current feature values.
        :return: state
        """
        
        return np.hstack(
            (self.velocity, self.acceleration, self.ego_desired_acceleration, self.gear, self.ego_desired_acceleration_previous, self.gear_previous, self.ego_vehicle.mass))
      
    # Intelligent driver model for ego desired acceleration    
    def driver_model_IDM(self):
      """
      Calculate the desired acceleration for car following model IDM
      :return: ego_desired_acceleration
      """
      #distance_init = 125  Remove as we aren't using radar

      rel_dist, rel_vel = self.get_obj_vehicle()
      approach_rate = self.velocity - self.obj_velocity
      # If crash, move leading car to 100 m ahead
      if(rel_dist < 0):
        print('CRASH: reset object' + str(rel_dist))
        self.obj_position = self.position + 100
        rel_dist = 100
     
      max_acc = 2.2
      comf_decel = 4
      accel_exp = 4
      s1 = 0
      
      if self.stoch:      
        if self.time< self.dt + .1:   # Only change driver at each drive cycle
            self.driver_noise = np.random.normal(0,.25, 1)
      
      else:
        self.driver_noise = [0]
        
      # Add stochasticity to driver
      s0 = 6 + 2 * self.driver_noise[0]  # Don't want to make smaller or will crash  (5-7)
      t_headway = 3.5 + self.driver_noise[0]   # (3 -4)         
      s_star = s0 + (approach_rate*self.velocity)/(2 * np.sqrt(max_acc*comf_decel)) + 1*s1 + t_headway*self.velocity
      s_star = (s_star / rel_dist)**2

      acc = 1 - (self.velocity/self.ego_desired_velocity)**accel_exp  - s_star 
      acc = acc*max_acc
      acc = np.clip(acc, -3.5, 3)
      return acc

    # Radar sensor for object vehicle (input: object_vehicle as VehicleModel)
    def get_obj_vehicle(self):
        ''' 
        Gets the velocity of the object vehicle from the drive cycle
        :return: relative disance, relative velocity (obj - ego)
        '''
        self.obj_velocity = self.get_drivecycle_velocity()
        self.obj_position += self.obj_velocity * self.dt
        relative_distance = self.obj_position - self.position
        relative_velocity = self.obj_velocity - self.velocity
        return relative_distance, relative_velocity
    
    def seed(self, seed=None):
        seed = np.random.randint(0, 100)
        return [seed]

    # Desired Velocity from Drive Cycle for leading vehicle
    def get_drivecycle_velocity(self):
        """
        Get the velocity from the loaded drive cycle
        """

        self.desired_velocity = self.drive_cycle_f(self.time)
        
        if self.stoch:
          if round(self.time, 1) % 180 == 0:   # Every 1 minutes
            #self.velocity_noise = np.random.normal(0,3,1)
            self.velocity_noise = self.rng.random() * 3
            #print(f' T {self.time} Vd {self.desired_velocity}   Noise {self.velocity_noise}')
          self.desired_velocity  = np.clip(self.desired_velocity + self.velocity_noise ,0, 38)
          
          return self.desired_velocity
        else:
          return self.desired_velocity


    def feature_scaling(self, state):
        """
        Min-Max-Scaler: scale X' = (X-Xmin) / (Xmax-Xmin)
        :param state:
        :return: scaled state
        """
        return (state - self.state_min) / (self.state_max - self.state_min)

    # Not used
    def reset_viewer(self):
        if self.viewer is not None:
            for key in self.viewer.history:
                self.viewer.history[key] = []
            self.viewer.components['signs'] = []

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

