import pybullet as p
import os
import math


class Goal:
    def __init__(self, client, base):
        f_name = os.path.join(os.path.dirname(__file__), 'simplegoal.urdf')
        self.client = client
        self.base = base
        self.goal = client.loadURDF(fileName=f_name, basePosition=[base[0], base[1], 0])

    def delete(self):
        try:
            self.client.removeBody(self.goal)
            self.goal = None  # Set to None to indicate the goal has been deleted
        except Exception:
            return
        
    def get_observation(self):
        # Get the position and orientation of the car in the simulation
        pos = self.base
        pos = pos[:2]
        # Concatenate position
        observation = pos

        return observation