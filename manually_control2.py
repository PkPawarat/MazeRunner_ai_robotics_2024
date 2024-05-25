import sys
import gym
import simple_driving
import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os
from pynput.keyboard import Key, Listener

sys.path.append("./simple-car-env-template")

# Test trained policy
env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=True, isDiscrete=True)
state = env.reset()[0]

# Setting up a listener for keyboard events
key_pressed = None

def on_press(key):
    global key_pressed
    try:
        if key.char in ['w', 'a', 's', 'd', 'z', 'c', 'x', 'q']:
            key_pressed = key.char
    except AttributeError:
        pass

def get_action():
    global key_pressed
    if key_pressed == 'w':
        return 7  # Forward
    elif key_pressed == 'a':
        return 8  # Forward-left
    elif key_pressed == 's':
        return 1  # Reverse
    elif key_pressed == 'd':
        return 6  # Forward-right
    elif key_pressed == 'z':
        return 0  # Reverse-left
    elif key_pressed == 'c':
        return 2  # Reverse-right
    elif key_pressed == 'x':
        return 4  # Reverse-right
    elif key_pressed == 'q':
        return 'q'  # Quit
    return None

listener = Listener(on_press=on_press)
listener.start()

listaction = []

while True:
    print("""Action Space:
                w: Forward
                a: Steer-left
                s: Reverse
                d: Steer-right
                Press 'q' to quit""")
    action = get_action()
    if action == 'q':
        break
    if action is not None:
        listaction.append(action)
        state_, reward, done, info, _ = env.step(action)
        state = state_
        env.render()

    time.sleep(0.1)  # Delay to allow for keyboard input handling

listener.stop()

for action in listaction:
    print(action)

env.close()
