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
import matplotlib as plt

# adapt as needed for your system
# from sys import platform
# if platform =="darwin":
#   matplotlib.use("Qt5Agg")
# else:
#   matplotlib.use('TkAgg')

from matplotlib import pyplot as plt

from env.hopf_network import HopfNetwork
from env.quadruped_gym_env import QuadrupedGymEnv


ADD_CARTESIAN_PD = True
TIME_STEP = 0.001
foot_y = 0.0838 # this is the hip length 
sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

env = QuadrupedGymEnv(render=True,              # visualize
                    on_rack=False,              # useful for debugging! 
                    isRLGymInterface=False,     # not using RL
                    time_step=TIME_STEP,
                    action_repeat=1,
                    motor_control_mode="TORQUE",
                    add_noise=False,    # start in ideal conditions
                    # record_video=True
                    )

# initialize Hopf Network, supply gait
cpg = HopfNetwork(time_step=TIME_STEP)

TEST_STEPS = int(10 / (TIME_STEP)/5)
t = np.arange(TEST_STEPS)*TIME_STEP

# [TODO] initialize data structures to save CPG and robot states
r = []
r_dot = []
theta = []
theta_dot = []
desired_foot_positions = []
actual_foot_positions = []
desired_joint_angles = []
actual_joint_angles = []
robot_base_velocities = []
energy = np.zeros((4, 3))
swing_durations = [[] for _ in range(4)]
stance_durations = [[] for _ in range(4)]
is_swing = [False] * 4  # Track if each leg is in swing phase
phase_duration = [0.0] * 4  # Accumulate time in the current phase for each leg


############## Sample Gains
# joint PD gains
kp=np.array([100,100,100])
kd=np.array([2,2,2])
# Cartesian PD gains
kpCartesian = np.diag([500]*3)
kdCartesian = np.diag([20]*3)

initial_x = env.robot.GetBasePosition()

for j in range(TEST_STEPS):
  # initialize torque array to send to motors
  
  action = np.zeros(12) 
  # get desired foot positions from CPG 
  xs,zs = cpg.update()

  #store cpg data
  r.append(cpg.get_r())
  r_dot.append(cpg.get_dr())
  curr_theta = cpg.get_theta()
  theta.append(curr_theta)
  theta_dot.append(cpg.get_dtheta())

  # [TODO] get current motor angles and velocities for joint PD, see GetMotorAngles(), GetMotorVelocities() in quadruped.py
  q = env.robot.GetMotorAngles().reshape((4,3))
  dq = env.robot.GetMotorVelocities().reshape((4,3))
  robot_base_velocities.append(env.robot.GetBaseLinearVelocity())

  # loop through desired foot positions and calculate torques
  for i in range(4):

    #verify if leg is in swing or stance phase
    if np.sin(curr_theta[i]) > 0:  # Swing phase
          if not is_swing[i]:  # Transition from stance to swing --> was not in stance phase at previous time step
              # End stance phase, save duration
              stance_durations[i].append(phase_duration[i])
              phase_duration[i] = 0.0
              is_swing[i] = True
          phase_duration[i] += TIME_STEP  # Accumulate swing time
    else:  # Stance phase
          if is_swing[i]:  # Transition from swing to stance
              # End swing phase, save duration
              swing_durations[i].append(phase_duration[i])
              phase_duration[i] = 0.0
              is_swing[i] = False
          phase_duration[i] += TIME_STEP  # Accumulate stance time

    # initialize torques for leg
    tau = np.zeros(3)
    # get desired foot i pos (xi, yi, zi) in leg frame
    leg_xyz = np.array([xs[i],sideSign[i] * foot_y,zs[i]])
    
    # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
    leg_q = env.robot.ComputeInverseKinematics(i, leg_xyz) #np.zeros(3) # [TODO] 

    # Add joint PD contribution to tau for leg i (Equation 4)
    tau += kp*(leg_q -q[i]) + kd*(0 -dq[i])#np.zeros(3) # [TODO] 

    # add Cartesian PD contribution
    if ADD_CARTESIAN_PD:
      # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
      J, pos = env.robot.ComputeJacobianAndPosition(i)# [TODO] 
      # Get current foot velocity in leg frame (Equation 2)
      v = J@dq[i]# [TODO] 
      # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
      tau += J.T @(kpCartesian@(leg_xyz-pos).T +kdCartesian@(0-v).T)#np.zeros(3) # [TODO]

    # Set tau for legi in action vector
    action[3*i:3*i+3] = tau

    #Store energy
    energy[i] += np.abs(tau * dq[i]) * TIME_STEP


    #store values
    if i == 1:
      desired_foot_positions.append(leg_xyz)
      desired_joint_angles.append(leg_q)
      actual_foot_positions.append(pos)
      actual_joint_angles.append(q[i])

  # send torques to robot and simulate TIME_STEP seconds 
  env.step(action) 

  # [TODO] save any CPG or robot states

#PROCESSING SIMULATION DATA
final_x = env.robot.GetBasePosition()
total_energy = np.sum(energy)  # Sum all energy used across joints
distance_traveled = np.sqrt((final_x[0] - initial_x[0])**2 + (final_x[1] - initial_x[1])**2+ (final_x[2] - initial_x[2])**2)  # Euclidean distance traveled
cot = total_energy / (np.sum(env.robot._total_mass_urdf) * 9.81 * distance_traveled)  # Cost of Transport
average_velocity = np.mean(robot_base_velocities) # Average velocity

for i in range(4):
    avg_swing_time = np.mean(swing_durations[i]) if swing_durations[i] else 0
    avg_stance_time = np.mean(stance_durations[i]) if stance_durations[i] else 0
    tot_swing_time = np.sum(swing_durations[i]) if swing_durations[i] else 0
    tot_stance_time = np.sum(stance_durations[i]) if stance_durations[i] else 0
    print(f"Leg {i+1}: Average Swing Time = {avg_swing_time:.3f}s, Average Stance Time = {avg_stance_time:.3f}s")
    print(f"Leg {i+1}: Average duty cycle =  {tot_stance_time/tot_swing_time:.3f}s")

print("Distance Traveled:", distance_traveled)
print("Cost of Transport (COT):", cot)
print("Average Velocity:", average_velocity)



##################################################### 
# PLOTS
#####################################################
# example
# fig = plt.figure()
# plt.plot(t,joint_pos[1,:], label='FR thigh')
# plt.legend()
# plt.show()


# Ensure two gait cycles (adjust based on your specific CPG settings)
num_cycles = 2
cycle_steps = int(TEST_STEPS / num_cycles)
t_cycle = t[:cycle_steps]

# Convert lists to arrays for easier slicing and plotting
r = np.array(r)
r_dot = np.array(r_dot)
theta = np.array(theta)
theta_dot = np.array(theta_dot)

# Define line properties
line_width = 1  # thinner lines for clarity
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']  # colors for different CPG states

# Plot CPG states for each leg
fig, axs = plt.subplots(4, 4, figsize=(15, 10))
fig.suptitle("CPG States (r, θ, ˙r, ˙θ) for Trot Gait", fontsize=16)

# Loop through each leg and plot the CPG states
for i in range(4):
    # Plot `r`
    axs[0, i].plot(t_cycle, r[:cycle_steps, i], color=colors[0], label="r", linewidth=line_width)
    axs[0, i].set_title(f"Leg {i+1} - r")
    axs[0, i].legend()

    # Plot `θ`
    axs[1, i].plot(t_cycle, theta[:cycle_steps, i], color=colors[1], label="θ", linewidth=line_width)
    axs[1, i].set_title(f"Leg {i+1} - θ")
    axs[1, i].legend()

    # Plot `˙r`
    axs[2, i].plot(t_cycle, r_dot[:cycle_steps, i], color=colors[2], label="˙r", linewidth=line_width)
    axs[2, i].set_title(f"Leg {i+1} - ˙r")
    axs[2, i].legend()

    # Plot `˙θ`
    axs[3, i].plot(t_cycle, theta_dot[:cycle_steps, i], color=colors[3], label="˙θ", linewidth=line_width)
    axs[3, i].set_title(f"Leg {i+1} - ˙θ")
    axs[3, i].legend()

# Adjust layout and display
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("plotsCPGstates.png")
plt.show()

# Plot for desired vs actual foot positions and joint angles for leg 1
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Desired vs Actual Foot Positions and Joint Angles (Leg 1)", fontsize=16)

# Define line colors for desired vs actual
desired_color = 'tab:blue'
actual_color = 'tab:orange'

desired_foot_positions = np.array(desired_foot_positions)
actual_foot_positions = np.array(actual_foot_positions)
desired_joint_angles = np.array(desired_joint_angles)
actual_joint_angles = np.array(actual_joint_angles)

# Foot positions for leg 1 (x, y, z)
axs[0, 0].plot(t, desired_foot_positions[:, 0], color=desired_color, label="Desired X", linewidth=line_width)
axs[0, 0].plot(t, actual_foot_positions[:, 0], color=actual_color, linestyle='--', label="Actual X", linewidth=line_width)
axs[0, 0].set_title("Foot Position X (Leg 1)")
axs[0, 0].legend()

axs[0, 1].plot(t, desired_foot_positions[:, 1], color=desired_color, label="Desired Y", linewidth=line_width)
axs[0, 1].plot(t, actual_foot_positions[:, 1], color=actual_color, linestyle='--', label="Actual Y", linewidth=line_width)
axs[0, 1].set_title("Foot Position Y (Leg 1)")
axs[0, 1].legend()

axs[0, 2].plot(t, desired_foot_positions[:, 2], color=desired_color, label="Desired Z", linewidth=line_width)
axs[0, 2].plot(t, actual_foot_positions[:, 2], color=actual_color, linestyle='--', label="Actual Z", linewidth=line_width)
axs[0, 2].set_title("Foot Position Z (Leg 1)")
axs[0, 2].legend()

# Joint angles for leg 1
axs[1, 0].plot(t, desired_joint_angles[:, 0], color=desired_color, label="Desired Angle 1", linewidth=line_width)
axs[1, 0].plot(t, actual_joint_angles[:, 0], color=actual_color, linestyle='--', label="Actual Angle 1", linewidth=line_width)
axs[1, 0].set_title("Joint Angle 1 (Leg 1)")
axs[1, 0].legend()

axs[1, 1].plot(t, desired_joint_angles[:, 1], color=desired_color, label="Desired Angle 2", linewidth=line_width)
axs[1, 1].plot(t, actual_joint_angles[:, 1], color=actual_color, linestyle='--', label="Actual Angle 2", linewidth=line_width)
axs[1, 1].set_title("Joint Angle 2 (Leg 1)")
axs[1, 1].legend()

axs[1, 2].plot(t, desired_joint_angles[:, 2], color=desired_color, label="Desired Angle 3", linewidth=line_width)
axs[1, 2].plot(t, actual_joint_angles[:, 2], color=actual_color, linestyle='--', label="Actual Angle 3", linewidth=line_width)
axs[1, 2].set_title("Joint Angle 3 (Leg 1)")
axs[1, 2].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("plotsCPGtracking.png")
plt.show()



