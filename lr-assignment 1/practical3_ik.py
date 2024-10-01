# Inverse Kinematics practical
from env.leg_gym_env import LegGymEnv
import numpy as np
from practical2_jacobian import jacobian_rel


def pseudoInverse(A,lam=0.001):
    """ Pseudo inverse of matrix A. 
        Make sure to take into account dimensions of A
            i.e. if A is mxn, what dimensions should pseudoInv(A) be if m>n 
        Also take into account potential singularities
    """
    m,n = np.shape(A)
    if m >= n:
        pinvA = np.linalg.inv((A.T @ A) + (lam**2  * np.eye(n))) @ A.T
    else:
        pinvA = A.T @ np.linalg.inv((A @ A.T) + (lam**2 * np.eye(m)))
    return pinvA

def ik_geometrical(xz,angleMode="<",l1=0.209,l2=0.195):
    """ Inverse kinematics based on geometrical reasoning.
        Input: Desired foot xz position (array) 
               angleMode (whether leg should look like > or <) 
               link lengths
        return: joint angles
    """
    x = xz[0]
    z = xz[1]
    q = np.zeros(2)
    
    if angleMode == "<":
        q[1] = - np.arccos( (x**2 + z**2 - l1**2 - l2**2) / (2 * l1 * l2))
    else:
        q[1] = np.arccos( (x**2 + z**2 - l1**2 - l2**2) / (2 * l1 * l2))
    q[0] = np.arctan2(-x,-z) - np.arctan( (l2 * np.sin(q[1])) / (l1 + l2 * np.cos(q[1])) )


    return q

def ik_numerical(q0,des_x,tol=1e-4):
    """ Numerical inverse kinematics
        Input: initial joint angle guess, desired end effector, tolerance
        return: joint angles
    """
    i = 0
    max_i = 100 # max iterations
    alpha = 0.5 # convergence factor
    lam = 0.001 # damping factor for pseudoInverse
    joint_angles = q0

    # Condition to iterate: while fewer than max iterations, and while error is greater than tolerance
    while( i < max_i and abs(sum(jacobian_rel(joint_angles)[1] - des_x)) > tol ):
        # Evaluate Jacobian based on current joint angles
        J, ee = jacobian_rel(joint_angles)

        # Compute pseudoinverse
        J_pinv = pseudoInverse(J, lam)

        # Find end effector error vector
        ee_error = des_x - ee

        # update joint_angles
        joint_angles += alpha * (J_pinv @ ee_error)

        # update iteration counter
        i += 1

    return joint_angles


if __name__ == "__main__":
        
    env = LegGymEnv(render=True, 
                    on_rack=True,    # set True to debug 
                    motor_control_mode='TORQUE',
                    action_repeat=1,
                    )

    NUM_STEPS = 5*1000   # simulate 5 seconds (sim dt is 0.001)
    tau = np.zeros(2) # either torques or motor angles, depending on mode

    IK_mode = "GEOMETRICAL"

    # sample joint PD gains
    kpJoint = np.array([55,55])
    kdJoint = np.array([0.8,0.8])

    # desired foot position (sample)
    des_foot_pos = np.array([0.1,-0.2])  ###0.1, -0.2


    #NEW
    T, FOOT_POS, FOOT_VEL, JOINT_POS, JOINT_VEL, JOINT_TAU = [], [], [], [], [], []
    COMMAND_POS = []

    for counter in range(NUM_STEPS):
        # Compute inverse kinematics in leg frame 
        if IK_mode == "GEOMETRICAL":
            # geometrical
            qdes = ik_geometrical(des_foot_pos, angleMode="<")


        else:
            # numerical
            qdes = ik_numerical(env.robot.GetMotorAngles(),des_foot_pos)
        
        # print 
        if counter % 500 == 0:
            J, ee_pos_legFrame = jacobian_rel(env.robot.GetMotorAngles())
            print('---------------', counter)
            print('q ik: ',qdes,'q real: ',env.robot.GetMotorAngles())
            print('ee pos: ', ee_pos_legFrame)

        # determine torque with joint PD
        tau = kpJoint*(qdes - env.robot.GetMotorAngles()) + kdJoint*(0 - env.robot.GetMotorVelocities())
        

        # add gravity compensation (Force), (get mass with env.robot.total_mass)
        if not env._on_rack:
            tau+= np.array([0,env.robot.total_mass*(-9.81)])

  
        # apply control, simulate
        env.step(tau)
