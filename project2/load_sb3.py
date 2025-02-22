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

import os, sys
import gym
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from sys import platform
# may be helpful depending on your system
# if platform =="darwin": # mac
#   import PyQt5
#   matplotlib.use("Qt5Agg")
# else: # linux
#   matplotlib.use('TkAgg')

# stable-baselines3
from stable_baselines3.common.monitor import load_results 
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, SAC
# from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.env_util import make_vec_env # fix for newer versions of stable-baselines3

from env.quadruped_gym_env import QuadrupedGymEnv
# utils
from utils.utils import plot_results
from utils.file_utils import get_latest_model, load_all_results


LEARNING_ALG = "PPO"
interm_dir = "./logs/intermediate_models/"
# path to saved models, i.e. interm_dir + '102824115106'
log_dir = interm_dir + '121624100040'

# initialize env configs (render at test time)
# check ideal conditions, as well as robustness to UNSEEN noise during training
env_config = {}
env_config['motor_control_mode'] = "CPG"
env_config['task_env'] = "FWD_LOCOMOTION"
env_config['observation_space_mode'] = "LR_COURSE_MEDIUM"
env_config['terrain'] = "STAIRS"
env_config['render'] = True
env_config['record_video'] = True
env_config['add_noise'] = True 
# env_config['competition_env'] = True

# get latest model and normalization stats, and plot 
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)
monitor_results = load_results(log_dir)
print(monitor_results)
plot_results([log_dir] , 10e10, 'timesteps', LEARNING_ALG + ' ')
plt.show() 

# reconstruct env 
env = lambda: QuadrupedGymEnv(**env_config)
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_path, env)
env.training = False    # do not update stats at test time
env.norm_reward = False # reward normalization is not needed at test time

# load model
if LEARNING_ALG == "PPO":
    model = PPO.load(model_name, env)
elif LEARNING_ALG == "SAC":
    model = SAC.load(model_name, env)
print("\nLoaded model", model_name, "\n")

obs = env.reset()
episode_reward = 0

# [TODO] initialize arrays to save data from simulation 
base_velocities = []  # List to store base velocity data
env_base_velocities = []  # List to store base velocity data from env
timestamps = []  
i_prev = 0
swing_durations = [[] for _ in range(4)]
stance_durations = [[] for _ in range(4)]
is_swing = [False] * 4  # Track if each leg is in swing phase
phase_duration = [0.0] * 4  # Accumulate time in the current phase for each leg
total_distance = 0.0
energy = 0.0

for j in range(7000):
    action, _states = model.predict(obs,deterministic=False) # sample at test time? ([TODO]: test)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards

    curr_theta = env.envs[0].env._cpg.get_theta()
    tau = env.envs[0].env.ScaleActionToCPGStateModulations(action[0])
    dq = env.envs[0].env.robot.GetMotorVelocities()
    energy += np.sum(np.abs(dq) * tau) * env.envs[0].env._time_step

    for i in range(4):

        #verify if leg is in swing or stance phase
        if np.sin(curr_theta[i]) > 0:  # Swing phase
            if not is_swing[i]:  # Transition from stance to swing --> was not in stance phase at previous time step
                # End stance phase, save duration
                stance_durations[i].append(phase_duration[i])
                phase_duration[i] = 0.0
                is_swing[i] = True
            phase_duration[i] += env.envs[0].env._time_step # Accumulate swing time
        else:  # Stance phase
            if is_swing[i]:  # Transition from swing to stance
                # End swing phase, save duration
                swing_durations[i].append(phase_duration[i])
                phase_duration[i] = 0.0
                is_swing[i] = False
            phase_duration[i] += env.envs[0].env._time_step  # Accumulate stance time

       

    if dones:
        print('episode_reward', episode_reward)
        print('Final base position', info[0]['base_pos'])
        print('Steps until failure', j-i_prev)
        i_prev = j
        episode_reward = 0
        env_base_velocities = []
        base_velocities.append(env_base_velocities)
        final_pos = env.envs[0].env.robot.GetBasePosition()
        distance = np.linalg.norm(np.array(final_pos))
        total_distance+=distance


    # [TODO] save data from current robot states for plots 
    # To get base position, for example: env.envs[0].env.robot.GetBasePosition() 
    base_velocity = (env.envs[0].env.robot.GetBaseLinearVelocity()[0]) 
    env_base_velocities.append(base_velocity)
   
    timestamps.append(i)
    
# [TODO] make plots:
#You will have simulated multiple times, select an i index to plot data from that simulation
all_velocities = np.concatenate([np.array(vel) for vel in base_velocities])
average_velocity = np.mean(all_velocities)
print("Average base velocity: ", average_velocity)
#repeat the average velocity for the length of the velocities array

# average_velocities = np.repeat(average_velocity, len(velocities))

# plt.figure()
# plt.plot(velocities, label="X-velocity", color='orange', linewidth=1.5) 
# plt.plot(average_velocities, label="Average X-velocity", color='red', linewidth=1.5) 
# plt.xlabel('Timestep')
# plt.ylabel('Base Velocity (m/s)')
# plt.title('Base Velocity Over Time')
# plt.legend()
# plt.show()

cot = energy / (np.sum(env.envs[0].robot._total_mass_urdf) * 9.81 * total_distance)  # Cost of Transport
print("Cost of Transport (COT):", cot)

for i in range(4):
    avg_swing_time = np.mean(swing_durations[i]) if swing_durations[i] else 0
    avg_stance_time = np.mean(stance_durations[i]) if stance_durations[i] else 0
    tot_swing_time = np.sum(swing_durations[i]) if swing_durations[i] else 0
    tot_stance_time = np.sum(stance_durations[i]) if stance_durations[i] else 0
    print(f"Leg {i+1}: Average Swing Time = {avg_swing_time:.3f}s, Average Stance Time = {avg_stance_time:.3f}s")
    print(f"Leg {i+1}: Average duty cycle =  {tot_stance_time/(tot_swing_time+tot_stance_time):.3f}s")



