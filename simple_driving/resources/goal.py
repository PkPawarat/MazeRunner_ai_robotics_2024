import pybullet as p
import os


class Goal:
    def __init__(self, client, base):
        f_name = os.path.join(os.path.dirname(__file__), 'simplegoal.urdf')
        self.client = client
        self.base = base
        self.goal = client.loadURDF(fileName=f_name, basePosition=[base[0], base[1], 0])

    def delete(self):
        self.client.removeBody(self.goal)
        self.goal = None  # Set to None to indicate the goal has been deleted