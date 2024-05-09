# CentralizedSharedLearning
A shared reinforcement learning algorithm for a fleet of vehicles serving a distribution of routes in a centralized framework.
The publication for this work can be found in Applied Energy (https://doi.org/10.1016/j.apenergy.2024.123217).

## Required Libraries
Python, Pytorch, Ray.io, OpenAI Gym

### Instructions
Create empty folders data, models, and agent_models. Set up the fleet of vehicles in main_ray.py and run this file. Data is saved as .mat files. The maps and settings in the specs file are arbitrary, they are not from a real vehicle. Add a folder with your own drive cycles and link to that folder in the DeterministicTrackClass.py.
