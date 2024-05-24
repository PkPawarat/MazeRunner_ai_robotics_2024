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
import pathplanning 
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
        self.observation_space = gym.spaces.Box(
            low=np.array([-80, -80, -80, -80, -80, -80, -80, -80, -80, -80, 0], dtype=np.float32),
            high=np.array([80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 200], dtype=np.float32))

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
        self.goal_objects = []
        self.goal = None
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        
        self.walls = [] # list of walls 
        self.maze = None # list of walls 
        self.maze_map = None # list of walls 
        
        self.current_goal = None # current goal object
        self.path_planner = None 
        self.start_node = None # start node
        self.end_node = None # end node
        self.shortest_path = None # shortest path
        self._envStepCounter = 0

        # New reward design parameters
        self.goal_reward = 50  # Reward for reaching the goal
        self.time_penalty = -0.1  # Penalty for each time step
        self.collision_penalty = -10  # Penalty for collisions
        self.distance_reward = 0.05  # Reward for covering distance towards the goal
        self.smooth_driving_reward = 0.01  # Reward for smooth driving behavior
        self.obstacle_avoidance_reward = 1  # Reward for avoiding obstacles
        self.max_distance_reward = 2  # Maximum reward for covering distance
        
        self._p.resetSimulation()
        self._p.setTimeStep(self._timeStep)
        self._p.setGravity(0, 0, -10)
        

        
    def step(self, action):
        # Feed action to the car and get observation of car's state
        reward = 0
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
                time.sleep(max(self._timeStep / 10, 0.01))  # Reduced sleep time

            if self._termination():
                self.done = True
                reward = -50
                break
            self._envStepCounter += 1
        
        closest_wall_pos, closest_wall_distance, rewardWall = self.closestWall()
        carpos = self.car.get_observation()
        
        # Compute reward
        reward += self.time_penalty
        dist_to_goal = self.get_dist_to_goal(carpos, self.get_goal_observation())
        reward += self.distance_reward * (self.prev_dist_to_goal - dist_to_goal)
        reward += self.smooth_driving_reward if action in [1, 7] else 0  # Smooth driving reward only forwards and backwards
        # reward += self.smooth_driving_reward if action in [3, 5, 4] else 0  # Smooth driving reward
        reward += self.obstacle_avoidance_reward if rewardWall == 0 else 0  # Obstacle avoidance reward
        reward += self.collision_penalty if rewardWall < 0 else 0  # Collision penalty

        # Update distance to goal
        self.prev_dist_to_goal = dist_to_goal

        # Check for goal reached
        if dist_to_goal < 1.2 and not self.reached_goal:
            reward += self.goal_reward
            if self.current_goal == len(self.goal_objects) - 1:
                self.done = True
                self.reached_goal = True
                reward += 500  # Additional reward for reaching last goal
                print("reached last goal____________Current goal -> Total goals:", self.current_goal, "->", len(self.goal_objects)-1)
            else: 
                self.goal_objects[self.current_goal].delete()
                self.current_goal += 1
                self.prev_dist_to_goal = self.get_dist_to_goal(carpos, self.get_goal_observation())
        
        # Get observation
        ob = self.get_observation(carpos, self.get_goal_observation(), closest_wall_pos)

        return ob, reward, self.done, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.current_goal = 0 # current goal object
        self.setupMaze()
        carpos = self.car.get_observation()
        self._envStepCounter = 0
        self.done = False
        self.reached_goal = False
        # Get observation to return
        self.prev_dist_to_goal = self.get_dist_to_goal(carpos, self.get_goal_observation())
        ob = self.get_observation(carpos, self.get_goal_observation())

        return ob

    def setupMaze(self):
        self.current_goal = 0 # current goal object
        self.start_node = None # start node
        self.end_node = None # end node

        listofNodes = []
        if self.path_planner is None:
            Plane(self._p)
            self.path_planner = pathplanning.PathPlanning()
        
            # Visual maze element in the environment
            self.maze = MazeClass("./_all-mazes/maze25x25s1.txt")
            self.maze_map = self.maze.readMazeFile()
            
            max_x = max([coord[0] for coord in self.maze_map.keys()]) + 1
            max_y = max([coord[1] for coord in self.maze_map.keys()]) + 1
            for y in range(max_y):
                for x in range(max_x):
                    position = (x-(max_x/2), y-(max_y/2))
                    node = pathplanning.Node(x=int(position[0]), y=int(position[1]))
                    if self.maze_map[(x, y)] == self.maze.WALL:
                        self.walls.append(Wall(self._p, position))
                        
                    elif self.maze_map[(x,y)] == self.maze.START:
                        # Visual element of the goal
                        self.car = Car(self._p, position)
                        self.start_node = node  # Define your start node
                    
                    elif self.maze_map[(x,y)] == self.maze.EXIT:
                        # Visual element of the goal
                        self.end_node = node  # Define your end node
                        listofNodes.append(node)
                        
                    elif self.maze_map[(x,y)] == self.maze.EMPTY:
                        listofNodes.append(node)
            
            self.path_planner.initial_path_planning(listofNodes)
            self.shortest_path = self.path_planner.execution(self.start_node, self.end_node)

            # Plot shortest_path
            carpos = self.car.get_observation()
            for node in self.shortest_path:
                dist = self.get_dist_to_goal(carpos, (node.X, node.Y))
                if dist <= 2:
                    pass
                else:
                    self.goal_objects.append(Goal(self._p, (node.X, node.Y)))

        else:
            self.car.reset()
        
        if self._renders:
            for index, obj in enumerate(self.goal_objects):
                if obj.goal is None:
                    self.goal_objects[index] = Goal(self._p, obj.base)        
                    
        self._envStepCounter = 0
        self.done = False
        self.reached_goal = False


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
                                                                    distance=40.0,
                                                                    yaw=90.0,
                                                                    pitch=-95,
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
        carpos, carorn = self._p.getBasePositionAndOrientation(self.car.car)
        goalpos, goalorn = self._p.getBasePositionAndOrientation(self.goal_objects[self.current_goal].goal)
        invCarPos, invCarOrn = self._p.invertTransform(carpos, carorn)
        goalPosInCar, goalOrnInCar = self._p.multiplyTransforms(invCarPos, invCarOrn, goalpos, goalorn)

        observation = [goalPosInCar[0], goalPosInCar[1]]
        # observation = [carpos[0], carpos[1]]
        return observation
    
    def get_observation(self, carpos, goal_pos, closest_wall_pos=None):
        if closest_wall_pos is None: 
            closest_wall_pos, _, _ = self.closestWall()
        
        return np.concatenate((carpos, goal_pos, closest_wall_pos, [self.current_goal]))
    
    def get_goal_observation(self):
        return self.goal_objects[self.current_goal].get_observation()

    def get_dist_to_goal(self, carpos, goal_pos):
        return math.sqrt(((carpos[0] - goal_pos[0]) ** 2 + (carpos[1] - goal_pos[1]) ** 2))

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

        # Calculate car orientation in Euler angles
        car_euler = self._p.getEulerFromQuaternion(carorn)
        car_heading = car_euler[2]

        # Define car's dimensions for calculating the points
        car_length = 0.6  # Length of the car
        car_width = 0.5   # Width of the car

        # Define four points on the car representing its corners
        car_points = [
            (carpos[0] + (car_length / 2) * math.cos(car_heading) + (car_width / 2) * math.sin(car_heading),
             carpos[1] + (car_length / 2) * math.sin(car_heading) - (car_width / 2) * math.cos(car_heading)),  # Front right
            (carpos[0] + (car_length / 2) * math.cos(car_heading) - (car_width / 2) * math.sin(car_heading),
             carpos[1] + (car_length / 2) * math.sin(car_heading) + (car_width / 2) * math.cos(car_heading)),  # Front left
            (carpos[0] - (car_length / 2) * math.cos(car_heading) + (car_width / 2) * math.sin(car_heading),
             carpos[1] - (car_length / 2) * math.sin(car_heading) - (car_width / 2) * math.cos(car_heading)),  # Back right
            (carpos[0] - (car_length / 2) * math.cos(car_heading) - (car_width / 2) * math.sin(car_heading),
             carpos[1] - (car_length / 2) * math.sin(car_heading) + (car_width / 2) * math.cos(car_heading))   # Back left
        ]

        closest_wall_distance = float('inf')  # Initialize minimum distance to positive infinity
        closest_wall_pos = None  # Initialize closest wall to None
        reward = 0

        for wall in self.walls:
            wall_pos, _ = self._p.getBasePositionAndOrientation(wall.wall)

            for point in car_points:
                # Calculate distance between each car point and the wall
                distance = math.sqrt((point[0] - wall_pos[0]) ** 2 + (point[1] - wall_pos[1]) ** 2)

                if distance < closest_wall_distance:
                    closest_wall_distance = distance
                    closest_wall_pos = wall_pos[:2]

                    if closest_wall_distance <= 0.50:  # Check if car is too close to the wall
                        # print("It's hitting the fucking wall!!!!!")
                        reward = -10    # punish if it hitting the wall
                        return closest_wall_pos, closest_wall_distance, reward

        return closest_wall_pos, closest_wall_distance, reward

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
        return self._envStepCounter > 5000

    def close(self):
        self._p.disconnect()
    
