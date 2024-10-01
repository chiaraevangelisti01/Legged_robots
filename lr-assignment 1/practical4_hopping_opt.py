# Hopping practical optimization
import numpy as np

from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES

from env.leg_gym_env import LegGymEnv
from practical2_jacobian import jacobian_rel
from practical3_ik import ik_numerical, ik_geometrical

SINGLE_JUMP = False

class HoppingProblem(ElementwiseProblem):
    """Define interface to problem (see pymoo documentation). """
    def __init__(self):
        super().__init__(n_var=2,                 # number of variables to optimize (sample)
                         n_obj=1,                 # number of objectives
                         n_ieq_constr=0,          # no inequalities 
                         xl=np.array([0.8, 50]),   # variable lower limits (what makes sense?)
                         xu=np.array([1.5, 200]))   # variable upper limits (what makes sense?) 
        # Define environment
        self.env = LegGymEnv(render=False,  # don't render during optimization
                on_rack=False, 
                motor_control_mode='TORQUE',
                action_repeat=1,
                )


    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate environment with variables chosen by optimization. """

        NUM_STEPS = 1000

        # Reset the environment before applying new profile (we want to start from the same conditions)
        self.env.reset()

        # Sample variables to optimize 
        f = x[0]        # hopping frequency
        Fz_max = x[1]   # max peak force in Z direction
        Fx_max = 0      # max peak force in X direction (can add)
        # [TODO] feel free to add more variables! What else could you optimize? 

        # Note: the below should look essentially the same as in practical4_hopping.py. 
        #   If you have some specific gains (or other variables here), make sure to test 
        #   the optimized variables under the same conditions.
        
        NUM_SECONDS = 5   # simulate N seconds (sim dt is 0.001)
        t = np.linspace(0,NUM_SECONDS,NUM_SECONDS*1000 + 1)

        # design Z force trajectory as a funtion of Fz_max, f, t
        #   Hint: use a sine function (but don't forget to remove positive forces)
        force_traj_z = np.zeros(len(t))
        force_traj_z = Fz_max*np.sin(2 * np.pi * f * t)
        # print(force_traj_z)
        force_traj_z[force_traj_z > 0] = 0

        # design X force trajectory as a funtion of Fx_max, f, t
        force_traj_x = np.zeros(len(t))
        force_traj_x = Fx_max * np.sin(2 * np.pi * f * t)
        force_traj_x[force_traj_x > 0] = 0

        if SINGLE_JUMP:
            # remove rest of profile (just keep the first peak)
            T = int(1000/f)
            force_traj_z[T:] = 0
            force_traj_x[T:] = 0
        
        # sample Cartesian PD gains (can change or optimize)
        kpCartesian = np.diag([500,300])
        kdCartesian = np.diag([30,20])

        # sample joint PD gains
        kpJoint = np.array([50,50])
        kdJoint = np.array([3.5,3.5])

        # sample nominal foot position (can change or optimize)
        nominal_foot_pos = np.array([0.0,-0.2]) 

        # Keep track of environment states - what should you optimize? how about for max lateral jumping?
        #   sample states to consider 
        sum_z_height = 0
        max_base_z = 0
        total_x_deviation = 0
          

        # Track the profile: what kind of controller will you use? 
        for i in range(NUM_SECONDS*1000):
            # Torques
            tau = np.zeros(2) 

            # Compute jacobian and foot_pos in leg frame (use GetMotorAngles() )
            J, ee_pos_legFrame = jacobian_rel(self.env.robot.GetMotorAngles())

            motor_vel = self.env.robot.GetMotorVelocities()
            foot_vel = J@motor_vel

            # Add Cartesian PD (and/or joint PD? Think carefully about this, and try it out.)
            tau += np.zeros(2) # [TODO]
            tau_cart = J.T@(kpCartesian@(nominal_foot_pos-ee_pos_legFrame).T + kdCartesian@(0-foot_vel).T)

            """INVERSE KINEMATICS"""
            #IF THE LEG IS IN THE AIR, APPLY INVERSE KINEMATICS TO FIX LEG SHAPE, ELSE DONT
            base_pos = self.env.robot.GetBasePosition()

            x_deviation = base_pos[0]
            total_x_deviation += x_deviation

            ground_contact = base_pos[2] + ee_pos_legFrame[1]

            if ground_contact > 0.04:
                # qdes = ik_numerical(env.robot.GetMotorAngles(),des_foot_pos)
                qdes = ik_geometrical(nominal_foot_pos, angleMode="<")
                tau_joint = kpJoint*(qdes - self.env.robot.GetMotorAngles()) + kdJoint*(0 - self.env.robot.GetMotorVelocities())
            else:
                tau_joint = 0

            # Add force profile contribution
            tau += tau_cart + tau_joint

            # Add gravity compensation (Force)
            tau += J.T@np.array([0,self.env.robot.total_mass*(-9.81)])

            # Add force profile contribution
            tau += J.T @ np.array([force_traj_x[i], force_traj_z[i]])

            # Apply control, simulate
            self.env.step(tau)

            # Record max base position (and/or other states)
            sum_z_height += base_pos[2]
            if base_pos[2] > max_base_z:
                max_base_z = base_pos[2]

        # objective function (what do we want to minimize?) COST FUNCTION
        #f1 = -max_base_z / Fz_max # TODO
        f1 = total_x_deviation

        out["F"] = [f1]


if __name__ == "__main__":
        
    # Define problem
    problem = HoppingProblem()

    # Define algorithms and initial conditions (depends on your variable ranges you selected above!)
    algorithm = CMAES(x0=np.array([1, 85])) # TODO: change initial conditions

    # Run optimization
    res = minimize(problem,
                algorithm,
                ('n_iter', 10), # may need to increase number of iterations
                seed=1,
                verbose=True)

    print(f"Best solution found: \nX = {res.X}\nF = {res.F}\nCV= {res.CV}")
    print("Check your optimized variables in practical4_hopping.py")