# LR-Leg
This repository contains an environment and exercises for simulating a quadruped robot leg.

## Installation

Recommend using a virtualenv (or conda) with python3.6 or higher. After installing virtualenv with pip, this can be done as follows:

`virtualenv venv_name --python=python3`

To activate the virtualenv: 

`source PATH_TO_VENV/bin/activate` 

Your command prompt should now look like: 

`(venv_name) user@pc:path$`

The repository depends on recent versions of pybullet, gym, numpy, pymoo, etc. as described on moodle. You can also install these with:

`pip install -r requirements.txt`


## Code structure

- [env](./env) for the leg environment files, please see the gym simulation environment [leg_gym_env.py](./env/leg_gym_env.py), the robot specific functionalities in [leg.py](./env/leg.py), and config variables in [configs_leg.py](./env/configs_leg.py). Review [leg.py](./env/leg.py) carefully for accessing robot states. 


## Code resources
- The [PyBullet Quickstart Guide](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.2ye70wns7io3) is the current up-to-date documentation for interfacing with the simulation. 
- The repository structure is inspired from [Google's motion_imitation project](https://github.com/erwincoumans/motion_imitation).

## Tips
- If your simulation is very slow, remove the calls to time.sleep() and disable the camera resets in [leg_gym_env.py](./env/leg_gym_env.py).
- The camera viewer can be modified in `_render_step_helper()` in [leg_gym_env.py](./env/leg_gym_env.py) to track the hopper.