import pybullet as p
import os

class Wall:
    def __init__(self, client, base):
        f_name = os.path.join(os.path.dirname(__file__), 'Parts & urdf/wall_v1/urdf/wall_v1.urdf')
        self.wall = client.loadURDF(fileName=f_name,
                   basePosition=[base[0], base[1], 0])

