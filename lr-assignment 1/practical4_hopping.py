# Hopping practical
from env.leg_gym_env import LegGymEnv
import numpy as np
import matplotlib.pyplot as plt
from practical2_jacobian import jacobian_rel
from practical3_ik import ik_numerical, ik_geometrical

env = LegGymEnv(render=True, 
                on_rack=False,    # set True to debug 
                motor_control_mode='TORQUE',
                action_repeat=1,
                record_video=False
                )

NUM_SECONDS = 3   # simulate N seconds (sim dt is 0.001)
tau_cart = np.zeros(2) # either torques or motor angles, depending on mode
NUM_STEPS = 1000

# peform one jump, or continuous jumping
SINGLE_JUMP = False

# sample Cartesian PD gains (can change or optimize)
kpCartesian = np.diag([500,300]) #initial 500,300  ##WORKS 200,300
kdCartesian = np.diag([30,20]) #initial 30,20       ##20,30

"""ADDING INVERSE KINEMATICS TRIAL"""
# sample joint PD gains
kpJoint = np.array([50,50])

kdJoint = np.array([3.50,3.50])

tau_joint = np.zeros(2)
l1=0.209
l2=0.195 
des_foot_pos = np.array([0,-0.2])


#define variables
joint_angles = env.robot.GetMotorAngles()
mass = env.robot.total_mass
joint_vel = env.robot.GetMotorVelocities()
J, foot_pos = jacobian_rel(joint_angles)
foot_vel=J@joint_vel

# define variables and force profile
NUM_STEPS = 1000
t = np.linspace(0,NUM_SECONDS,NUM_SECONDS*NUM_STEPS + 1)


if SINGLE_JUMP:
    # may want to choose different parameters
    Fz_max = 10 * env.robot.total_mass*9.81     # max peak force in Z direction -> factor >1 * gravity
    Fx_max = 0     # max peak force in X direction
    f = 1.2
else: ### CONTINUOUS JUMP
    Fz_max = 5 * env.robot.total_mass*9.81     # max peak force in Z direction
    Fx_max = 0.35 * Fz_max
    f = 1.2    # frequency

# design Z force trajectory as a funtion of Fz_max, f, t
#   Hint: use a sine function (but don't forget to remove positive forces)
force_traj_z = np.zeros(len(t))
force_traj_z = Fz_max*np.sin(2 * np.pi * f * t)
force_traj_z[force_traj_z > 0] = 0

# design X force trajectory as a funtion of Fx_max, f, t
force_traj_x = np.zeros(len(t))
force_traj_x = Fx_max * np.sin(2 * np.pi * f * t)
force_traj_x[force_traj_x > 0] = 0


if SINGLE_JUMP:
    # remove rest of profile (just keep the first peak)
    T = int(NUM_STEPS/f)
    force_traj_z[T:] = 0
    force_traj_x[T:] = 0

# sample nominal foot position (can change or optimize)
nominal_foot_pos = np.array([0.0,-0.2]) 

# keep track of max z height
max_base_z = 0

# Track the profile: what kind of controller will you use? 

#definition of large matrices to store data (for plotting)
foot_pos_tot = np.zeros((NUM_SECONDS*NUM_STEPS + 1,2))
#foot_vel_tot = np.zeros((NUM_SECONDS*NUM_STEPS + 1,2))
torque_tot = np.zeros((NUM_SECONDS*NUM_STEPS + 1,2))
total_forces = np.zeros((NUM_SECONDS*NUM_STEPS + 1,2)) 

for i in range(NUM_SECONDS*NUM_STEPS):
    # Torques
    tau = np.zeros(2) ##REINITIALIZE TAU BEFORE EACH INTERATION

    # Compute jacobian and foot_pos in leg frame (use GetMotorAngles() )
    J, ee_pos_legFrame = jacobian_rel(env.robot.GetMotorAngles())

    # Get foot velocity in leg frame (use GetMotorVelocities() )
    motor_vel = env.robot.GetMotorVelocities()
    foot_vel = J@motor_vel

    # Add Cartesian PD (and/or joint PD? Think carefully about this, and try it out.)
    tau_cart = J.T@(kpCartesian@(nominal_foot_pos-ee_pos_legFrame).T + kdCartesian@(0-foot_vel).T)


    """INVERSE KINEMATICS"""
    #IF THE LEG IS IN THE AIR, APPLY INVERSE KINEMATICS TO FIX LEG SHAPE, ELSE DONT
    base_pos = env.robot.GetBasePosition()
    ground_contact = base_pos[2] + ee_pos_legFrame[1]

    if ground_contact > 0.04:
        # qdes = ik_numerical(env.robot.GetMotorAngles(),des_foot_pos)
        qdes = ik_geometrical(des_foot_pos, angleMode="<")
        tau_joint = kpJoint*(qdes - env.robot.GetMotorAngles()) + kdJoint*(0 - env.robot.GetMotorVelocities())
    else:
        tau_joint = 0

    # Add force profile contribution
    tau += tau_cart + tau_joint + J.T @ np.array([force_traj_x[i], force_traj_z[i]])

    # Add gravity compensation (Force)
    tau += J.T@np.array([0,env.robot.total_mass*(-9.81)])

    # Apply control, simulate
    env.step(tau)

    # Record max base position (and/or other states)
    if max_base_z < base_pos[2]:
        max_base_z = base_pos[2]

    foot_pos_tot[i] = ee_pos_legFrame
    torque_tot[i] = tau
    total_forces[i] = np.array([force_traj_x[i], force_traj_z[i]])

print('Peak z', max_base_z)

# [TODO] make some plots to verify your force profile and system states
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2,2, figsize=(15,10))


axs[0, 0].plot(t, foot_pos_tot.T[0], color = 'blue', label = 'foot position x')
axs[0, 0].plot(t, nominal_foot_pos[0]*np.ones(NUM_SECONDS*NUM_STEPS + 1), color = 'green', linestyle = 'dashed', label = 'desired foot position x')
axs[0, 0].plot(t, foot_pos_tot.T[1], color = 'red', label = 'foot position y')
axs[0, 0].plot(t, nominal_foot_pos[1]*np.ones(NUM_SECONDS*NUM_STEPS + 1), color = 'yellow', linestyle = 'dashed', label = 'desired foot position y')
axs[0, 0].set_title('Position and velocity of the foot in (x,y)')
axs[0, 0].set_ylabel('position [m]')

# plot Forces
axs[0, 1].plot(t, total_forces.T[0], color = 'blue', label = 'Force Fx')
axs[0, 1].plot(t, total_forces.T[1], color = 'red', label = 'Force Fz')
axs[0, 1].set_title('Forces Fx and Fz')
axs[0, 1].set_ylabel('Forces [N]')

# plot torques
axs[1, 0].plot(t, torque_tot.T[0], color = 'blue', label = 'torque tau1')
axs[1, 0].plot(t, torque_tot.T[1], color = 'red', label = 'torque tau2')
axs[1, 0].set_title('Torques tau1 and tau2')
axs[1, 0].set_ylabel('Torque [Nm]')

#setting the legends
axs[0, 0].legend()
axs[0, 0].set_xlabel('time [s]')
axs[0, 1].legend()
axs[0, 1].set_xlabel('time [s]')
axs[1, 0].legend()
axs[1, 0].set_xlabel('time [s]')

axs[1, 1].axis('off')

plt.subplots_adjust(hspace=0.3)

print('Program finished successfully!')

fig.savefig('plots_4-1.png')
