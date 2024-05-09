# -*- coding: utf-8 -*-
"""
Vehicle Model Class
- Specs and model of the AMT vehicle
Created on Wed Oct 27 13:15:37 2021

@author: linds
"""

import numpy as np
import scipy._lib
import scipy._lib.decorator
from scipy import interpolate, integrate
from control import forced_response, TransferFunction
import scipy.io as sio

G = 9.81  # gravity
F1 = 0.015  # rolling friction
RHO = 1.184  # density air, kg/m³
M_S_MPH = 0.44704  # 1 mph = 0.44704 m/s
KM_H_M_S = 3.6  # 1 m/s = 3.6 km/h
DIESEL_D = 0.835
CURR_DIR = '/scratch/lsutto2/AMT_Central_SubSize/'

class Vehicle_Model():
    def __init__(self, car_id=0):
        #super(Vehicle_Model, self).__init__()
        self.car_id = car_id
        self.get_specs()
        self.dwell = 0
        self.wheel_radius = float(self.specs['rw'])
        self.ratio_gear = np.array(self.specs['r_gear'], float)
        self.ratio_fd = float(self.specs['r_diff'])
        self.eff_fd = float(self.specs['eff_diff'])
        self.Wengine = 650
        self.prev_t_engine = 0
        self.prev_t_b = 0
        gear_path = r'{}specs_AMT/gear_eff.mat'.format(CURR_DIR)
        self.gear_eff_map = sio.loadmat(gear_path)
        mdot_path = r'{}specs_AMT/m_dot.mat'.format(CURR_DIR)
        self.mdot_map = sio.loadmat(mdot_path)
        
        self.mass = 16000
        self.idle_velocity = float(self.specs['min_We'])*(np.pi/30)/(self.ratio_fd*self.ratio_gear[0])*self.wheel_radius
        
    def set_base_mass(self, car_id):
        if np.ndim(car_id) > 0:
            self.mass = np.array([9070, 13000, 160000, 19000])
        else:
            if car_id == 0:
              self.mass = 9070
            elif car_id == 1:
              self.mass = 13000
            elif car_id == 2: 
              self.mass = 16000
            elif car_id == 3:
              self.mass = 19000

    
    def get_minTengine(self, We):
          minT_torque = np.array(self.specs['minT'])
          minT_engspd = np.array(self.specs['maxT_We'])
          t_min_f = interpolate.interp1d(minT_engspd,minT_torque)
          return t_min_f(We)

    def get_maxTengine(self, We):
          maxT_torque = np.array(self.specs['maxT'])
          maxT_engspd = np.array(self.specs['maxT_We'])
          t_max_f = interpolate.interp1d(maxT_engspd,maxT_torque)
          return t_max_f(We)

    def get_engineSpeed_in_gear(self, velocity, gear):

        if velocity <= self.idle_velocity:
          return float(self.specs['min_We'])

        # Get engine speed in RPM
        Wwheel = velocity/self.wheel_radius
        Wengine = Wwheel*(self.ratio_fd*self.ratio_gear[gear-1])*30/np.pi

        return Wengine

    # Adjust gear (if necessary) to keep engine speed within limits
    # return engine speed and gear                         
    def get_engineSpeed(self, velocity, gear):
        old_gear = gear
        # Check if stopped, idle in 1st gear
        # minimum vehicle speed to limit velocity to idle
        if velocity <= float(self.specs['min_We'])*(np.pi/30)/(self.ratio_fd*self.ratio_gear[0])*self.wheel_radius: # these are all fixed terms it may be clearer to turn this into a named variable
          return float(self.specs['min_We']), 1
        
        # Get engine speed in RPM
        Wwheel = velocity/self.wheel_radius
        Wengine = Wwheel*(self.ratio_fd*self.ratio_gear[gear-1])*30/np.pi       # this math is repeated a few times, maybe it should be a function call?
        WengineOrig = Wwheel*(self.ratio_fd*self.ratio_gear[gear-1])*30/np.pi 
        new_gear = gear
        # Requires downshift to avoid engine stall                              # is this rather niave shifting?
        if Wengine < float(self.specs['min_We']):                          # if engine is *close* to min speed, why the soft limit?
          new_gear = gear - 1
          gear = np.clip(new_gear, 1, 10)                                       # why clip? hasnt 1st gear idle already been checked? clip does a max which shouldn't be needed
          Wengine = Wwheel*(self.ratio_fd*self.ratio_gear[gear-1])*30/np.pi
          assert not new_gear==0, f'You undersped. Entered engine speed {WengineOrig}, gear {old_gear}, and velocity {velocity} /n Left engine speed {Wengine}, gear {gear}'
        # Requires upshift 
        elif Wengine > float(self.specs['max_We']):
          new_gear = gear + 1
          gear = np.clip(new_gear, 1, 10)

          Wengine = Wwheel*(self.ratio_fd*self.ratio_gear[gear-1])*30/np.pi
          assert not new_gear==11, f'You oversped. Entered with engine speed of {WengineOrig}, gear {old_gear}, and velocity {velocity} /n Left with engine speed of {Wengine}, gear {gear}'

        try:
            if (Wengine < float(self.specs['min_We'])) or (Wengine > float(self.specs['max_We'])):
                Wengine,gear = self.get_engineSpeed(velocity, gear)
        except:
            print(f'Entered engine speed {WengineOrig}, gear {old_gear}, and velocity {velocity} /n Left engine speed {Wengine}, gear {gear}')
            raise
        # Set alert if engine speed is outside limits 
        assert float(self.specs['min_We']) < Wengine < float(self.specs['max_We']),\
            f'Entered engine speed {WengineOrig}, gear {old_gear}, and velocity {velocity} /n Left engine speed {Wengine}, gear {gear}'
        
        # Return either original value or return new value and near gear
        return Wengine, gear

    # Matching Simulink Sim
    def get_inertia(self, gear):
        ''' Vehicle Inertia '''
        ''' mass rw**2 + Iw + (rg rfd)**2 eta Ie'''
        Ivm = (self.ratio_fd*self.ratio_gear[int(gear-1)])**2*float(self.specs['inertia_engine']) 
        Ivm += float(self.specs['inertia_wheel']) + float(self.mass)*self.wheel_radius**2
        return Ivm 

    def get_gear_efficiency(self, We, gear):
        ''' Gearbox Efficiency (AMT) '''
        ge_We = np.array([800,1000,1400,1900])
        ge_gear = np.array([1,2,3,4,5,6,7,8,9,10])
        gear_eff_f = interpolate.interp2d(ge_gear,ge_We, np.matrix(self.gear_eff_map['eff_gear']))
        gear_eff = gear_eff_f(gear-1, We)[0]
        return gear_eff


    # ADD ROAD GRADE HERE - always positive
    def get_resistive_torque(self, velocity):
        ''' Vehicle Resistive Torque '''
        grade = 0
        # Changed to not encourage positive torque at almost stopped for 0 a_des
        if velocity <= 0.3:    
          return 0  # Can't go backwards
        
        f_aero = .5*self.specs['Cd'] * self.specs['Af'] * RHO * velocity**2
        f_roll = self.mass*G*F1*np.cos(grade)
        #f_grade = self.mass*G*np.sin(grade)  # Grade
        f_grade = 0
        f_resistive = f_aero + f_roll + f_grade
        t_resistive = f_resistive*self.wheel_radius
        return t_resistive

    # Positive traction torque
    def get_engine_torque(self, tt_from_engine, velocity, gear):
        ''' Get the desired engine torque from the traction torque request '''

        # Returns new gear if necessary
        We, gear = self.get_engineSpeed(velocity, gear)   
        t_engine = tt_from_engine/(self.ratio_fd*self.ratio_gear[gear-1])
        t_engine_max = self.get_maxTengine(We)
        # Cannot go over torque limit
        
        # Attempt at adding a time delay
        num = [1]
        dem = [0.2, 1]
        tf_s = TransferFunction(num, dem)        
        T, t_engine = forced_response(tf_s, [0, 0.2, 0.4, 0.6, .8], [self.prev_t_engine, self.prev_t_engine, t_engine, t_engine, 0], 0 )
        t_engine = t_engine[3]

        t_engine = np.minimum(t_engine, t_engine_max)
        #self.prev_t_engine = t_engine
        return t_engine, gear

    # Get fuel rate from engine operating points
    def get_fuelrate(self, velocity, eng_torque, gear, dt):
      """ Lookup table for fuel rate 
          :param: engine speed RPM
          :param: engine torque NM
          :return: fuel rate
      """

      # Should keep current gear at this point
      rpm, ng = self.get_engineSpeed(velocity, gear)

      mdot_torque = np.array(self.specs['mdot_T'])
      mdot_engspd = np.array(self.specs['mdot_We'])
      fuel_rate_f = interpolate.interp2d(mdot_engspd, mdot_torque, np.matrix(self.mdot_map['m_dot']))
      fuel_rate = fuel_rate_f(rpm, eng_torque)

       # Set alert if gear changed
      assert ng == gear,\
            f'{rpm} in gear {gear} ({ng}) invalid bounds'
      
      return fuel_rate[0]

    # Get the limits on actual braking torque - Negative Traction Torque
    def get_actual_brake_torque(self, T_traction, velocity, gear):

      w_engine, gear = self.get_engineSpeed(velocity, gear)
      
      T_engineBrake = self.get_minTengine(w_engine)
      T_engineBrake = np.maximum(T_engineBrake, T_traction/(self.ratio_fd*self.ratio_gear[gear-1]))  # Negative torque
      T_serviceBrake = T_traction - T_engineBrake*(self.ratio_fd*self.ratio_gear[gear-1])  # Remaining torque to service brake
      max_brake = -G*float(self.mass) * float(self.specs['b_cr']) * float(self.specs['rw'])

      #num = [1]
      #dem = [0.2, 1]
      #tf_s = TransferFunction(num, dem)

      #T, T_serviceBrake = forced_response(tf_s, [0, 0.2, 0.4, 0.6, .8], [self.prev_t_b, self.prev_t_b, T_serviceBrake, T_serviceBrake, 0] )
      #T_serviceBrake = T_serviceBrake[4]
      
      T_serviceBrake = np.maximum(T_serviceBrake, max_brake)

      #self.prev_t_b = T_serviceBrake
      return T_serviceBrake, T_engineBrake, gear

    # Get actual acceleration from traction torque
    def calc_acceleration_from_tt(self, velocity, Tt_e, Tb, gear):
        """
        Calculates the corresonding acceleration for a specific
        traction torque at a specific velocity.
        :param vel: velocity
        :param Tt: traction torque to the wheel
        :return: acceleration
        """

        We, _ = self.get_engineSpeed(velocity, gear)
        Ivm = self.get_inertia(gear)
        t_resistive = self.get_resistive_torque(velocity)
        acceleration = (Tt_e - t_resistive + Tb)/Ivm * self.wheel_radius
        
        return acceleration

    def get_remaining_torque_toWheel(self, Tengine, velocity, gear, t_control):
        """
        Calculates the engine torque required for a given traction torque
        :param Tt: traction torque to the wheel
        :return: Torque to wheel
        """
        if velocity <= 0:
          return Tengine*(self.ratio_fd*self.ratio_gear[gear-1])

        We, _ = self.get_engineSpeed(velocity, gear)
        We_rads = We*np.pi/30  # rad/s
        
        # Calulcate the Auxilliary torques
        t_fan = np.maximum(0, 0.0052*We_rads**2+0.0013*We_rads-0.0764)    
        t_alt = 1901/We_rads
        t_other = 22.72

        # Torque going to the transmission (removed efficiencies for now)
        t_to_trans = max(0,Tengine - t_fan - t_alt - t_other)
        gear_eff = self.get_gear_efficiency(We, gear)/100
        eff_fd = self.eff_fd/100

        t_to_wheel_from_engine = t_to_trans*(self.ratio_fd*self.ratio_gear[gear-1])
        if t_to_wheel_from_engine >= 0:
          t_to_wheel_from_engine = t_to_wheel_from_engine*gear_eff*eff_fd
        else:
          t_to_wheel_from_engine = t_to_wheel_from_engine/gear_eff/eff_fd
       
        return t_to_wheel_from_engine


    def get_limited_braking(self, T_traction, velocity, gear, dt):
        We, gear = self.get_engineSpeed(velocity, gear)
        Ivm = self.get_inertia(gear)
        t_resistive = self.get_resistive_torque(velocity)
        T_traction_lim = -velocity*Ivm/(dt* self.wheel_radius) + t_resistive
        T_traction_lim = min(T_traction_lim, 0)  # Added to make sure it doesn't go positive
        T_traction = max(T_traction_lim, T_traction)

        return T_traction, gear

    def get_Ttraction_from_desiredAccel(self, d_accel, velocity, gear):
        We, _ = self.get_engineSpeed(velocity, gear)
        Ivm = self.get_inertia(gear)
        T_resistive = self.get_resistive_torque(velocity)
        T_traction = d_accel * Ivm / self.wheel_radius + T_resistive

        return T_traction
  

    def select_fuel_optimal_gear(self, velocity, torque, gear):

      velocity = np.maximum(.001, velocity)
      agear = gear - 1
      rpm, _ = self.get_engineSpeed(velocity, gear)  # RPM

      delta_p= np.maximum(0, (-2/5*velocity+10)*1000)

      mdot_torque = np.array(self.specs['mdot_T'])
      mdot_engspd = np.array(self.specs['mdot_We'])
      mdot_f = interpolate.interp2d(mdot_engspd, mdot_torque, np.matrix(self.mdot_map['m_dot']))

      maxT_torque = np.array(self.specs['maxT'])
      maxT_engspd = np.array(self.specs['maxT_We'])
      t_max_f  = interpolate.interp1d(maxT_engspd,maxT_torque)

      index = np.zeros(len(self.ratio_gear))  # index[0] = gear 1
      rpm_in_gear = np.zeros(len(self.ratio_gear))
      torque_in_gear = np.zeros(len(self.ratio_gear))
      # Set the index 
      if torque > 0:
        while np.all((index == 0)):
          for g in range(len(self.ratio_gear)):  # 0 to 9

            rpm_in_gear[g] = np.maximum(650, velocity/self.wheel_radius*self.ratio_gear[g]*self.ratio_fd*30/np.pi)

            torque_in_gear[g] = torque/(self.ratio_gear[g]*self.ratio_fd)
            # Check if operating point is viable
            if rpm_in_gear[g] >= 650 and rpm_in_gear[g] <= 1900:
              if rpm_in_gear[g]/rpm > 2/3 and rpm_in_gear[g]/rpm < 1.5:
                #torque_in_gear[g] = torque/(self.ratio_gear[g]*self.ratio_fd)
                t_max = t_max_f(rpm_in_gear[g])
                #print('tmax ' + str(t_max))
                if torque_in_gear[g] < (t_max-delta_p/(rpm_in_gear[g]*np.pi/30)):
                  index[g]=1         
                  
          if rpm_in_gear[0]<650:
            index[0]=1;
          torque=torque-5000/velocity;
      else: # negative torque
        for g in range(len(self.ratio_gear)):
          #rpm_in_gear[g] = np.maximum(650, velocity/self.wheel_radius*self.ratio_gear[g]*self.ratio_fd*30/np.pi)
          rpm_in_gear[g] = velocity/self.wheel_radius*self.ratio_gear[g]*self.ratio_fd*30/np.pi
          torque_in_gear[g] = torque/(self.ratio_gear[g]*self.ratio_fd)
          if 650 <= rpm_in_gear[g] and rpm_in_gear[g] <= 1800:
            if rpm_in_gear[g]/rpm > 2/3 and rpm_in_gear[g]/rpm < 1.5:
              #print('rpm gear ' + str(rpm_in_gear[g]) + ' g ' + str(g))
              torque_min = self.get_minTengine(rpm_in_gear[g])
              #print('t_min ' + str(torque_min))
              #print(' torque in ' + str(g) + ' ' + str(torque_in_gear[g]))
              if torque_in_gear[g] > torque_min:
                index[g] = -1
                

      for g in range(len(self.ratio_gear)):
        if index[g] < 0:
          index[g] = -1
        elif index[g] > 0:
          index[g] = mdot_f(rpm_in_gear[g],torque_in_gear[g])
        else:  # index == 0
          index[g] = 9999

      
      # Choose best gear via index
      y = agear
      min=9999  
      for g in range(len(self.ratio_gear)):
        if index[g] > 0:
          index[g] = index[g] + 0.22 * (self.dwell + abs(g-agear))
          if index[g] < min:
            min=index[g]
            y = g
          if rpm >= 1800 and y == agear:
            if y < 9:
              y += 1
            else:
              print('vehicle unable to find a gear')
        elif index[g] < 0:
          y = g
          break
      u = y - agear
      self.dwell += np.abs(u)
      agear += u
      gear = agear + 1
      return gear


    def get_specs(self):
        """
        set vehicle specs to default
        """
        # TODO: add more cars!
        #if self.car_id == 'truck':
        self.specs = {
                'trans_type': 'AMT',
        
                'Af': 7.71,
                'Cd': 0.8,
                'acceleration_limits': [-3, 3],
                'velocity_limits': [0, 37],
                'power_limits': [-250, 250],
                'b_cr': 0.3, 
                'rw': .498, 
                'r_diff': 2.64,
                'eff_diff':93.67,
                'r_gear' : [7.3978,5.4415,4.2470,3.4286,2.9388,2.1577,1.5871,1.2387,1,0.8571],
                'inertia_engine': 4.141,
                'inertia_wheel' : 183.0651,
                'maxT_We' : [650,700,800,900,1000,1100,1200,1300,1400,1550,1600,1700,1800,1900,2000,2100,2150],
                'maxT' : [1582,1666.2,1827.8,2230.4,2267.8,2255,2243.4,2240.6,2237.6,2074.9,2053.6,1788.2,1245.2,870.8,530.3,0,0],
                'minT' : [-124.5,-126.7,-132.7,-135.9,-142,-150,-159.2,-169.3,-180.2,-193,-204.7,-220,-237,-254.9,-274.2,-296.2,-317.2], 
                'mdot_T' : [-250,-200,-150,0,260,520,780,1040,1300,1560,1820,2077,2250],
                'mdot_We' : [650,779,906,1033,1161,1289,1417,1544,1672,1800],
                'maxP' : 380000,
                'max_We' : 2050,
                'min_We' : 650

            }

        #else:
        #    raise NotImplementedError('Not a truck')



