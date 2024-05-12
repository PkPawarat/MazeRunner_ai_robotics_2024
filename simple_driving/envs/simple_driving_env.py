import gym
import numpy as np
import math
import pybullet as p
from pybullet_utils import bullet_client as bc
from simple_driving.resources.car import Car
from simple_driving.resources.plane import Plane
from simple_driving.resources.goal import Goal
from simple_driving.resources.obstacle import Obstacle
from simple_driving.resources.wall import Wall
from maze import MazeClass
import matplotlib.pyplot as plt
import time
import os

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class SimpleDrivingEnv(gym.Env):
    metadata = {'render.modes': ['human', 'fp_camera', 'tp_camera']}

    def __init__(self, isDiscrete=True, renders=False):
        if (isDiscrete):
            self.action_space = gym.spaces.Discrete(9)
        else:
            self.action_space = gym.spaces.box.Box(
                low=np.array([-1, -.6], dtype=np.float32),
                high=np.array([1, .6], dtype=np.float32))
        # self.observation_space = gym.spaces.box.Box(
        #     low=np.array([-40, -40, -40, -40, -40, -40, -40, -40, -40, -40], dtype=np.float32),
        #     high=np.array([40, 40, -40, -40, -40, -40, -40, -40, -40, -40], dtype=np.float32))
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-80, -80], dtype=np.float32),
            high=np.array([80, 80], dtype=np.float32))
        self.np_random, _ = gym.utils.seeding.np_random()

        if renders:
          self._p = bc.BulletClient(connection_mode=p.GUI)
        else:
          self._p = bc.BulletClient()

        self.reached_goal = False
        self._timeStep = 0.01
        self._actionRepeat = 50
        self._renders = renders
        self._isDiscrete = isDiscrete
        self.car = None
        self.goal_object = None
        self.goal = None
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        
        self.walls = [] # list of walls 
        self.maze = None # list of walls 
        
        self.listPos = [] # list of previously positions 
        # self.reset()
        self._envStepCounter = 0

    def step(self, action):
        # Feed action to the car and get observation of car's state
        if (self._isDiscrete):
            fwd = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
            steerings = [-0.6, 0, 0.6, -0.6, 0, 0.6, -0.6, 0, 0.6]
            throttle = fwd[action]
            steering_angle = steerings[action]
            action = [throttle, steering_angle]
        self.car.apply_action(action)
        for i in range(self._actionRepeat):
          self._p.stepSimulation()
          if self._renders:
            time.sleep(self._timeStep)
          
          
          # Simulate LiDAR
          lidar_data = self.simulate_lidar()
          #print("Lidar data:", lidar_data)
          carpos, carorn = self._p.getBasePositionAndOrientation(self.car.car)
          goalpos, goalorn = self._p.getBasePositionAndOrientation(self.goal_object.goal)
          
          car_ob = self.getExtendedObservation() + lidar_data  # Append LiDAR data to the car's observation
          
          if self._termination():
            self.done = True
            reward = -50
            break
          self._envStepCounter += 1

        # Compute reward as L2 change in distance to goal
        # dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
                                  # (car_ob[1] - self.goal[1]) ** 2))
        dist_to_goal = math.sqrt(((carpos[0] - goalpos[0]) ** 2 +
                                  (carpos[1] - goalpos[1]) ** 2))
        # reward = max(self.prev_dist_to_goal - dist_to_goal, 0)
        reward = -dist_to_goal
        self.prev_dist_to_goal = dist_to_goal
        
        # Done by reaching goal
        if dist_to_goal < 1.5 and not self.reached_goal:
            print("reached goal________________________________________________________________")
            self.done = True
            self.reached_goal = True
            reward += 50 # if it's reached goal add reward 50
            
        ob = car_ob
        
        # # Convert list positions to numpy arrays
        # if self._envStepCounter % 1000 == 0:
        #     ob_np = np.array(ob)
        #     listPos_np = np.array(self.listPos)

        #     # Calculate distances
        #     distances = np.linalg.norm(listPos_np - ob_np, axis=1)

        #     # Check if any position in listPos is within the range of 0.5 units close to car_ob
        #     is_close = any(distances < 0.05)
            
        #     # # Debugging: Print out the positions in listPos for easier debugging
        #     # print("Positions in listPos:", listPos_np)

        #     if is_close:
        #         reward += -50
        #         self.done = True
            
        #     self.listPos.append(car_ob)
            
        
        # closestwall, closestwallpos, rewardWall = self.closestWall()
        # if rewardWall < 0:
        #     print("hit wall")
        #     reward = -500
        #     self.done = True
        
        return ob, reward, self.done, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._p.resetSimulation()
        self._p.setTimeStep(self._timeStep)
        self._p.setGravity(0, 0, -10)
        # Reload the plane and car
        Plane(self._p)
        
        self.listPos.clear()
        # Visual maze element in the environment
        maze = MazeClass("./_all-mazes/maze25x25s2.txt")
        self.maze = maze.readMazeFile()
        max_x = max([coord[0] for coord in self.maze.keys()]) + 1
        max_y = max([coord[1] for coord in self.maze.keys()]) + 1
        # print("Maze X:", max_x)
        # print("Maze Y:", max_y)        
        for y in range(max_y):
            for x in range(max_x):
                if self.maze[(x, y)] == maze.WALL:
                    wall = (x-(max_x/2), y-(max_y/2))
                    self.walls.append(Wall(self._p, wall))
                    
                if self.maze[(x,y)] == maze.START:
                    start = (x-(max_x/2), y-(max_y/2))
                    self.car = Car(self._p, start)
                
                if self.maze[(x,y)] == maze.EXIT:
                    # Visual element of the goal
                    self.goal = (x-(max_x/2), y-(max_y/2))
                    self.goal_object = Goal(self._p, self.goal)

        self._envStepCounter = 0

        self.done = False
        self.reached_goal = False

        # Get observation to return
        carpos = self.car.get_observation()

        self.prev_dist_to_goal = math.sqrt(((carpos[0] - self.goal[0]) ** 2 +
                                           (carpos[1] - self.goal[1]) ** 2))
        
        car_ob = self.getExtendedObservation()
        self.listPos.append(car_ob)
        # Concatenate car's extended observation with the closest obstacle position
        ob = car_ob

        return ob

    def render(self, mode='human'):
        if mode == "fp_camera":
            # Base information
            car_id = self.car.get_ids()
            proj_matrix = self._p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                       nearVal=0.01, farVal=100)
            pos, ori = [list(l) for l in
                        self._p.getBasePositionAndOrientation(car_id)]
            pos[2] = 0.2

            # Rotate camera direction
            rot_mat = np.array(self._p.getMatrixFromQuaternion(ori)).reshape(3, 3)
            camera_vec = np.matmul(rot_mat, [1, 0, 0])
            up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
            view_matrix = self._p.computeViewMatrix(pos, pos + camera_vec, up_vec)

            # Display image
            # frame = self._p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
            # frame = np.reshape(frame, (100, 100, 4))
            (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                                      height=RENDER_HEIGHT,
                                                      viewMatrix=view_matrix,
                                                      projectionMatrix=proj_matrix,
                                                      renderer=p.ER_BULLET_HARDWARE_OPENGL)
            frame = np.array(px)
            frame = frame[:, :, :3]
            return frame
            # self.rendered_img.set_data(frame)
            # plt.draw()
            # plt.pause(.00001)

        elif mode == "tp_camera":
            car_id = self.car.get_ids()
            base_pos, orn = self._p.getBasePositionAndOrientation(car_id)
            view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                                    distance=20.0,
                                                                    yaw=40.0,
                                                                    pitch=-35,
                                                                    roll=0,
                                                                    upAxisIndex=2)
            proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                             aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                             nearVal=0.1,
                                                             farVal=100.0)
            (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                                      height=RENDER_HEIGHT,
                                                      viewMatrix=view_matrix,
                                                      projectionMatrix=proj_matrix,
                                                      renderer=p.ER_BULLET_HARDWARE_OPENGL)
            frame = np.array(px)
            frame = frame[:, :, :3]
            return frame
        else:
            return np.array([])

    def getExtendedObservation(self):
        # self._observation = []  #self._racecar.getObservation()
        carpos, carorn = self._p.getBasePositionAndOrientation(self.car.car)
        goalpos, goalorn = self._p.getBasePositionAndOrientation(self.goal_object.goal)
        invCarPos, invCarOrn = self._p.invertTransform(carpos, carorn)
        goalPosInCar, goalOrnInCar = self._p.multiplyTransforms(invCarPos, invCarOrn, goalpos, goalorn)

        observation = [goalPosInCar[0], goalPosInCar[1]]
        # observation = [carpos[0], carpos[1]]
        return observation
    
    def closestWall(self):
        """
        Finds the closest wall to the car in the simulation environment.

        This function assumes the following:
            - self._p refers to a physics engine client (e.g., PyBullet)
            - self.car.car is the name of the car object
            - self.walls is an iterable containing wall objects in the environment
            - Each wall object has a property named 'wall' that represents its visual or collision representation

        Returns:
            The closest wall object (or None if no walls are found)
        """

        carpos, carorn = self._p.getBasePositionAndOrientation(self.car.car)
        invCarPos, invCarOrn = self._p.invertTransform(carpos, carorn)

        # Define car's dimensions for calculating the points
        car_length = 0.6  # Length of the car
        car_width = 0.5   # Width of the car

        # Define four points on the car representing its corners
        car_points = [
            ((car_length / 2)+carpos[0], (car_width / 2)+carpos[1]),   # Front right
            ((car_length / 2)+carpos[0], -(car_width / 2)+carpos[1]),  # Front left
            (-(car_length / 2)+carpos[0], (car_width / 2)+carpos[1]),  # Back right
            (-(car_length / 2)+carpos[0], -(car_width / 2)+carpos[1])  # Back left
        ]

        closest_wall_pos = float('inf')  # Initialize minimum distance to positive infinity
        closest_wall = None  # Initialize closest wall to None
        reward = 0

        for wall in self.walls:
            wall_pos, _ = self._p.getBasePositionAndOrientation(wall.wall)

            # Check each car point against the wall
            for point in car_points:
                # Calculate distance between the point and the wall
                distance = math.sqrt((point[0] - wall_pos[0]) ** 2 + (point[1] - wall_pos[1]) ** 2)

                if distance < closest_wall_pos:
                    closest_wall_pos = distance
                    closest_wall = wall

                    if closest_wall_pos <= 0.6:
                        reward = -50
                        return closest_wall, closest_wall_pos, reward

        return closest_wall, closest_wall_pos, reward

    def simulate_lidar(self, num_rays=360, ray_length=10):
        lidar_link_index = self._p.getNumJoints(self.car.car) - 1
        lidar_pos, lidar_orn = self._p.getLinkState(self.car.car, lidar_link_index)[4:6]
        
        rays_from = []
        rays_to = []
        rotation_matrix = self._p.getMatrixFromQuaternion(lidar_orn)
        forward_vector = [rotation_matrix[0], rotation_matrix[3], rotation_matrix[6]]
        up_vector = [rotation_matrix[2], rotation_matrix[5], rotation_matrix[8]]
        right_vector = np.cross(forward_vector, up_vector)
        
        hit_distances = []
        for i in range(num_rays):
            angle = 2 * np.pi * i / num_rays
            direction = [
                forward_vector[0] * np.cos(angle) + right_vector[0] * np.sin(angle),
                forward_vector[1] * np.cos(angle) + right_vector[1] * np.sin(angle),
                forward_vector[2] * np.cos(angle) + right_vector[2] * np.sin(angle)
            ]
            ray_from = lidar_pos
            ray_to = [lidar_pos[0] + ray_length * direction[0], 
                    lidar_pos[1] + ray_length * direction[1], 
                    lidar_pos[2] + ray_length * direction[2]]
            rays_from.append(ray_from)
            rays_to.append(ray_to)

        results = self._p.rayTestBatch(rays_from, rays_to)
        for i, result in enumerate(results):
            hit_distance = result[2] * ray_length
            hit_distances.append(hit_distance)
            color = [1, 0, 0] if hit_distance < ray_length else [0, 1, 0]
            self._p.addUserDebugLine(ray_from, rays_to[i], color, lifeTime=0.1)

        return hit_distances
    def _termination(self):
        return self._envStepCounter > 10000

    def close(self):
        self._p.disconnect()
    
