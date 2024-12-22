# SPDX-FileCopyrightText: Copyright (c) 2022 Guillaume Bellegarda. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2022 EPFL, Guillaume Bellegarda

""" Run CPG """
import time
import numpy as np
import matplotlib

# adapt as needed for your system
# from sys import platform
# if platform =="darwin":
#   matplotlib.use("Qt5Agg")
# else:
#   matplotlib.use('TkAgg')

from matplotlib import pyplot as plt

from env.hopf_network import HopfNetwork
from env.quadruped_gym_env import QuadrupedGymEnv
from env.quadruped_motor import QuadrupedMotorModel

ADD_CARTESIAN_PD = True
TIME_STEP = 0.001                     # initially: 0.001
foot_y = 0.0838                       # this is the hip length 
sideSign = np.array([-1, 1, -1, 1])   # get correct hip sign (body right is negative)

env = QuadrupedGymEnv(render=True,              # visualize
                    on_rack=False,              # useful for debugging! 
                    isRLGymInterface=False,     # not using RL
                    time_step=TIME_STEP,
                    action_repeat=1,
                    motor_control_mode="TORQUE",
                    add_noise=False,            # start in ideal conditions
                    # record_video=True
                    )

# initialize Hopf Network, supply gait
gait = "WALK" #can also: TROT, PACE, WALK, BOUND
carac = "medium" #can also: slow, fast
cpg = HopfNetwork(time_step=TIME_STEP, gait=gait) #, mu=mu, omega_swing=omega_swing, omega_stance=omega_stance, gait=GAIT, alpha=alpha, coupling_strength=coupling_strength, couple=couple, ground_clearance=ground_clearance, ground_penetration=ground_penetration, robot_height=robot_height, des_step_len=des_step_len, max_step_len_rl=max_step_len_rl, use_RL=use_RL)

TEST_STEPS = int(3 / (TIME_STEP)) # number of steps to simulate
t = np.arange(TEST_STEPS)*TIME_STEP
print(t)

# [TODO] initialize data structures to save CPG and robot states
r                     = []
r_dot                 = []
theta                 = []
theta_dot             = []
desired_foot_position = []
actual_foot_position  = []
desired_joint_angle   = []
actual_joint_angle    = []
actual_joint_velocity = []
torques               = []
computed_torques      = []
body_velocities       = []
Energy_consumption    = []


############## Sample Gains ##############
# joint PD gains
kp=np.array([135]*3)
kd=np.array([1, 6, 8])
# Cartesian PD gains
kpCartesian = np.diag([600]*3)
kdCartesian = np.diag([20]*3)
##########################################

for j in range(TEST_STEPS):
  # initialize torque array to send to motors
  action = np.zeros(12) 
  # get desired foot positions from CPG 
  xs,zs = cpg.update()
  # [TODO] get current motor angles and velocities for joint PD, see GetMotorAngles(), GetMotorVelocities() in quadruped.py
  q  = env.robot.GetMotorAngles()
  dq = env.robot.GetMotorVelocities()
  #print('q:', q.shape, '\n', q, '\ndq:', dq.shape, '\n', dq)
  
  ##################
  #  store values  #
  ##################
  r.append(cpg.get_r())
  r_dot.append(cpg.get_dr())
  theta.append(cpg.get_theta())
  theta_dot.append(cpg.get_dtheta())
  #for plots

  #initalize leg values
  temp_desired_foot_position = []
  temp_actual_foot_position  = []
  temp_desired_joint_angle   = []

  # loop through desired foot positions and calculate torques
  for i in range(4):
    q_i  = q[3*i:3*i+3]
    dq_i = dq[3*i:3*i+3]
    # initialize torques for legi
    tau = np.zeros(3)
    # get desired foot i pos (xi, yi, zi) in leg frame
    leg_xyz = np.array([xs[i],sideSign[i] * foot_y,zs[i]])
    #print ('leg_xyz', i, leg_xyz)
    #if i == 1: print('leg_xyz:', leg_xyz)
    # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
    leg_q = env.robot.ComputeInverseKinematics(i ,leg_xyz) 
    
    # Add joint PD contribution to tau for leg i (Equation 4)
    tau += kp * (leg_q - q_i) + kd * (0 - dq_i) 

    # add Cartesian PD contribution
    if ADD_CARTESIAN_PD:
      # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
      Jac, pos = env.robot.ComputeJacobianAndPosition(i)
      # Get current foot velocity in leg frame (Equation 2)
      vel = Jac @ dq_i # [TODO:Done!]
      # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
      tau += Jac.T @ (kpCartesian@(leg_xyz - pos) + kdCartesian@( 0 - vel)) #size = 1x3
      #print('tau:', i, tau)

    # Set tau for legi in action vector
    action[3*i:3*i+3] = tau
    #print('action:', action.shape, action)
    
    
    temp_desired_foot_position.append(leg_xyz)              #Adds vector [x, y, z] of leg i to the list of the desired position of the foot
    temp_actual_foot_position.append(pos)                   #Adds vector [x, y, z] of leg i to the list of the actual position of the foot
    temp_desired_joint_angle.append(leg_q)                  #Adds vector [q1, q2, q3] of leg i to the list of the desired joint angles of the leg

  # send torques to robot and simulate TIME_STEP seconds 
  env.step(action)
  
  # Saving of CPG and robot states
  # save CPG and robot states
  desired_foot_position.append(temp_desired_foot_position)  #Adds list of desired foot positions of one timestep to the list of desired foot positions
  actual_foot_position.append(temp_actual_foot_position)    #Adds list of actual foot positions of one timestep to the list of actual foot positions
  desired_joint_angle.append(temp_desired_joint_angle)      #Adds list of desired joint angles of one timestep to the list of desired joint angles
  # q and dq have shape (12,) with 3 values for each leg i: [q1i, q2i, q3i, q1i+1, q2i+1, q3i+1, ...]
  actual_joint_angle.append(q)                              #Adds list of actual joint angles of one timestep to the list of actual joint angles
  actual_joint_velocity.append(dq)                          #Adds list of actual joint velocities of one timestep to the list of actual joint velocities
  # Save torque values: each timestep has 12 values for each motor of shape (12,) same way as q and dq
  torques.append(np.sum(np.abs(action)))                    #Adds list of torques of one timestep to the list of torques
  
  #Calculate torques from motor commands
  motor_torques = env.robot.GetMotorTorques()                       # Gets motor torques from motor commands
  Energy_consumption.append(np.abs(np.dot(action, dq))*TIME_STEP)   # Energy consumption of the robot is the dot product of the torque and the velocity at each timestep
  
  motor_torques = motor_torques.reshape(4, 3)
  computed_torques.append(np.sum(np.abs(motor_torques), axis=1))    #Adds list of computed torques of one timestep to the list of computed torques
  
  #Robot_saved.X = cpg.X.copy()         # [TODO] save CPG states
  #Robot_saved.robot = env.robot.copy() # [TODO] save any CPG or robot states
  BodyVel = env.robot.GetBaseLinearVelocity()
  Absolute_Vel = np.sqrt(BodyVel[0]**2 + BodyVel[1]**2 + BodyVel[2]**2)
  body_velocities.append([Absolute_Vel, BodyVel[0]])
  #Energy_consumption.append(np.dot(action, dq))                     # Energy consumption of the robot is the dot product of the torque and the velocity at each timestep
  

##################################################### 
#                      PLOTS                        #
#####################################################
# Convert lists to arrays for easier slicing and plotting
r                = np.array(r)
r_dot            = np.array(r_dot)
theta            = np.array(theta)
theta_dot        = np.array(theta_dot)
computed_torques = np.array(computed_torques)

#Calculation of duty cycle and time duration of one step for leg 1
touch_ground = (theta // np.pi).T[0]
#print("touch_ground", touch_ground, '\n', np.diff(touch_ground))
hit_ground = [i for i in range(1, len(touch_ground)) if touch_ground[i] == 1 and touch_ground[i-1] == 0]
lift_off   = [i for i in range(1, len(touch_ground)) if touch_ground[i] == 0 and touch_ground[i-1] == 1]

steps_one_cycle = (hit_ground[-1] - hit_ground[0])/(len(hit_ground)-1)
step_duration   = steps_one_cycle*TIME_STEP
duty_cycle      = 0
if touch_ground[0]   == 1: hit_ground = [0] + hit_ground
elif touch_ground[0] == 0: lift_off   = [0] + lift_off
print("hit_ground", hit_ground, '\n', "lift_off", lift_off)
duty_cycle = (lift_off[1] - hit_ground[0])/(hit_ground[1] - hit_ground[0])


# Ensure two gait cycles (adjust based on your specific CPG settings)
num_cycles  = 2
len_cycle_walk = 313 + 20
cycle_steps = len_cycle_walk*num_cycles       # number of steps in a defined number of cycles
t_cycle     = t[:cycle_steps]

# Define line properties
line_width = 1                                                   # thinner lines for clarity
colors     = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']  # colors for different CPG states

# print("computed_torques\n", computed_torques.shape, computed_torques)

# Plot CPG states for each leg
fig, axs = plt.subplots(4, 4, figsize=(15, 10))                  # 4x4 grid of subplots: columns for each leg, rows for each CPG state
title_string = r"CPG States $(r, θ, \dot r, \dot θ)$ for " + gait + " Gait"
fig.suptitle(title_string, fontsize=20)

# Loop through each leg and plot the CPG states
for i in range(4):
    # Plot ⁠ r ⁠
    axs[0, i].plot(t_cycle, r[:cycle_steps, i], color=colors[0], label="r", linewidth=line_width)
    axs[0, i].set_title(f"Leg {i+1} - r")
    axs[0, i].legend()

    # Plot ⁠ θ ⁠
    axs[1, i].plot(t_cycle, theta[:cycle_steps, i], color=colors[1], label="θ", linewidth=line_width)
    axs[1, i].set_title(f"Leg {i+1} - θ")
    axs[1, i].legend()

    # Plot ⁠ ˙r ⁠
    axs[2, i].plot(t_cycle, r_dot[:cycle_steps, i], color=colors[2], label=r"$\dot r$", linewidth=line_width)
    axs[2, i].set_title(f"Leg {i+1} - ˙r")
    axs[2, i].legend()

    # Plot ⁠ ˙θ ⁠
    axs[3, i].plot(t_cycle, theta_dot[:cycle_steps, i], color=colors[3], label=r"$\dot θ$", linewidth=line_width)
    axs[3, i].set_title(f"Leg {i+1} - ˙θ")
    axs[3, i].legend()
    
    #set labels
    if i == 0:
      axs[0, i].set_ylabel(r"$r$")
      axs[1, i].set_ylabel(r"$\theta [rad]$")
      axs[2, i].set_ylabel(r"$\dot r$")
      axs[3, i].set_ylabel(r"$\dot \theta [\frac{rad}{s}]$")
    
    axs[3, i].set_xlabel("Time [s]")

# Adjust layout and display
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f'states_CPG_Leg1_{gait}_{carac}.png')
plt.show()

# Plot for desired vs actual foot positions and joint angles for leg 1
fig, axs = plt.subplots(2, 3, figsize=(15, 8))          # 2x3 grid of subplots: 3 for foot [x, y, z] positions and 3 for joint angles
fig.suptitle("Desired vs Actual Foot Positions and Joint Angles (Leg 1)", fontsize=20)

# Define line colors for desired vs actual
desired_color = 'tab:blue'
actual_color = 'tab:orange'

desired_foot_positions = np.array(desired_foot_position)
actual_foot_positions  = np.array(actual_foot_position)
# q and dq have shape (12,) with 3 values for each leg i: [q1i, q2i, q3i, q1i+1, q2i+1, q3i+1, ...]
# Reshape to (4, 3, -1) to get 4 legs, 3 joint angles, for all time steps
desired_joint_angles   = np.array(desired_joint_angle).reshape(-1, 4, 3) 
actual_joint_angles    = np.array(actual_joint_angle).reshape(-1, 4, 3)
#print('desired_foot_positions', desired_foot_positions.shape, '\n', desired_foot_positions)
#print('desired_joint_angles', desired_joint_angles.shape, '\n', desired_joint_angles)

# Extract x, y, z positions for leg 0
x_positions_leg0 = [pos[0][0] for pos in desired_foot_positions]
y_positions_leg0 = [pos[0][1] for pos in desired_foot_positions]
z_positions_leg0 = [pos[0][2] for pos in desired_foot_positions]

# Extract x, y, z positions for leg 0 (actual positions)
x_positions_leg0_actual = [pos[0][0] for pos in actual_foot_positions]
y_positions_leg0_actual = [pos[0][1] for pos in actual_foot_positions]
z_positions_leg0_actual = [pos[0][2] for pos in actual_foot_positions]

# Plotting for leg 0's actual x, y, z positions alongside the desired ones
axs[0, 0].plot(t[:cycle_steps*2], x_positions_leg0_actual[:cycle_steps*2], color=actual_color, linestyle='--', label="Actual X (Leg 0)", linewidth=line_width)
axs[0, 1].plot(t[:cycle_steps*2], y_positions_leg0_actual[:cycle_steps*2], color=actual_color, linestyle='--', label="Actual Y (Leg 0)", linewidth=line_width)
axs[0, 2].plot(t[:cycle_steps*2], z_positions_leg0_actual[:cycle_steps*2], color=actual_color, linestyle='--', label="Actual Z (Leg 0)", linewidth=line_width)

# Plotting for leg 1's x, y, z positions
axs[0, 0].plot(t[:cycle_steps*2], x_positions_leg0[:cycle_steps*2], color=desired_color, label="Desired X (Leg 0)", linewidth=line_width)
axs[0, 1].plot(t[:cycle_steps*2], y_positions_leg0[:cycle_steps*2], color=desired_color, label="Desired Y (Leg 0)", linewidth=line_width)
axs[0, 2].plot(t[:cycle_steps*2], z_positions_leg0[:cycle_steps*2], color=desired_color, label="Desired Z (Leg 0)", linewidth=line_width)

# Titles and legends for each subplot
axs[0, 0].set_title("Foot Position X (Leg 0)")
axs[0, 0].legend()
axs[0, 1].set_title("Foot Position Y (Leg 0)")
axs[0, 1].legend()
axs[0, 2].set_title("Foot Position Z (Leg 0)")
axs[0, 2].legend()

joint1_leg0 = [pos[0][0] for pos in desired_joint_angles]
joint2_leg0 = [pos[0][1] for pos in desired_joint_angles]
joint3_leg0 = [pos[0][2] for pos in desired_joint_angles]

joint1_leg0_actual = [pos[0][0] for pos in actual_joint_angles]
joint2_leg0_actual = [pos[0][1] for pos in actual_joint_angles]
joint3_leg0_actual = [pos[0][2] for pos in actual_joint_angles]

axs[1, 0].plot(t[:cycle_steps*2], joint1_leg0[:cycle_steps*2], color=desired_color, label="Desired Joint 1 (Leg 0)", linewidth=line_width)
axs[1, 1].plot(t[:cycle_steps*2], joint2_leg0[:cycle_steps*2], color=desired_color, label="Desired Joint 2 (Leg 0)", linewidth=line_width)
axs[1, 2].plot(t[:cycle_steps*2], joint3_leg0[:cycle_steps*2], color=desired_color, label="Desired Joint 3 (Leg 0)", linewidth=line_width)

axs[1, 0].plot(t[:cycle_steps*2], joint1_leg0_actual[:cycle_steps*2], color=actual_color, linestyle='--', label="Actual Joint 1 (Leg 0)", linewidth=line_width)
axs[1, 1].plot(t[:cycle_steps*2], joint2_leg0_actual[:cycle_steps*2], color=actual_color, linestyle='--', label="Actual Joint 2 (Leg 0)", linewidth=line_width)
axs[1, 2].plot(t[:cycle_steps*2], joint3_leg0_actual[:cycle_steps*2], color=actual_color, linestyle='--', label="Actual Joint 3 (Leg 0)", linewidth=line_width)

# Joint angles for leg 1
axs[1, 0].set_title("Joint Angle 1 (Leg 1)")
axs[1, 0].legend()
axs[1, 1].set_title("Joint Angle 2 (Leg 1)")
axs[1, 1].legend()
axs[1, 2].set_title("Joint Angle 3 (Leg 1)")
axs[1, 2].legend()

# Set labels
axs[0, 0].set_ylabel("Position [m]")
axs[1, 0].set_ylabel("Angle [rad]")
axs[1, 0].set_xlabel("Time [s]")
axs[1, 1].set_xlabel("Time [s]")
axs[1, 2].set_xlabel("Time [s]")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f'Foot_Joint_Leg1_{gait}_{carac}.png')
plt.show()

#Calculate parameters needed
#cost of transport
print('Cost of transport:', np.sum(np.abs(torques)))
print('Computer Cost of transport:', np.sum(computed_torques))


#Body velocity
body_velocities = np.array(body_velocities)
BaseVel, XVel = body_velocities.T
mean_vel   = np.mean(BaseVel)
max_vel    = np.max(BaseVel)
min_vel    = np.min(BaseVel)
mean_x_vel = np.mean(XVel)
print("mean velocity of body", mean_vel)


filer = open(f"calculated_params_{gait}_{carac}", 'w')
filer.write(f"Cost of Transport = {np.sum(computed_torques)} [Nm]\n")
filer.write(f"Total Energy Consumption = {np.sum(Energy_consumption)} [J]\n")
filer.write(f'Duty cycle = {duty_cycle}\n')
filer.write(f'step duration = {step_duration}\n')
filer.write(f'Average body velocity is {mean_vel} [m/s] with x-component {mean_x_vel}\n with min at {min_vel} and max at {max_vel}\n')
filer.close
print("!!!File written!!!")