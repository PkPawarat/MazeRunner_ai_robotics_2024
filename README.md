# MazeRunner_ai_robotics_2024

## Project Description

MazeRunner is a project dedicated to implementing Deep Reinforcement Learning (DRL) for autonomous maze navigation. The project's primary objective is to develop an AI-driven robot capable of independently navigating through a maze to reach specific goals in a predetermined sequence.

### Goal

The overarching goal of MazeRunner is to engineer an autonomous robot proficient in navigating complex mazes, leveraging reinforcement learning techniques to identify and reach goals in the correct order.

### Approach

1. **Deep Reinforcement Learning (DRL)**:

   - Maze navigation will be achieved through DRL methodologies, enabling the robot to learn optimal navigation strategies through interaction with its environment.
   - Similar to our previous project, where a car navigated towards a goal, here the robot will maneuver through the maze towards the designated goals.

2. **Maze Environment**:

   - A simulated maze environment will be created, featuring various layouts with walls and obstacles.
   - Maze complexity will be varied to challenge the robot's navigational capabilities.

## Implementation

The MazeRunner implementation will encompass several key steps:

1. **Environment Setup**:

   - Creation of a simulated maze environment using appropriate software tools (e.g., Pygame, Unity, etc.).
   - Design and implementation of maze configurations, incorporating walls and obstacles.

2. **Deep Reinforcement Learning**:

   - Implementation of a DRL algorithm (e.g., Q-learning, Deep Q-Networks, etc.) to train the robot for maze navigation.
   - Fine-tuning of hyperparameters and training of the DRL model within the maze environment.

3. **Integration**:

   - Seamless integration of the DRL-based navigation system with the robot's sensor inputs for optimal navigation.
   - Rigorous testing of the integrated system within the maze environment to validate navigation capabilities.

## Usage

To deploy and utilize MazeRunner, adhere to the following steps:

1. **Setup Environment**:

   - Install requisite dependencies and libraries.
   - Execute "python .\setup.py install".
   - Utilize Python version 3.12.2.
   - Initialize the maze environment.

2. **Training**:

   - Initiate training of the DRL model by executing the designated training script.

3. **Testing**:
   - Conduct comprehensive testing of the trained model within the maze environment to verify navigation proficiency.

## Action Space

- 0: Reverse-Left
- 1: Reverse
- 2: Reverse-Right
- 3: Steer-Left (no throttle)
- 4: No throttle and no steering
- 5: Steer-Right (no throttle)
- 6: Forward-right
- 7: Forward
- 8: Forward-left

## Contributors

- Pawarat Phatthanaphusakun 13662352 - Lead Software Developer & Project Manager
- Laurentius Setiadharma 14018295 - Assistant Manager
- Zahead Rashidi 13366378 - Contributor
- Mostafa Rahimi 13914736 - Contributor
- Hamish McEwan 14397273 - Contributor

## Documentation

Detailed documentation for this project: [[Documentation]](https://pkpawarat.github.io/MazeRunner_ai_robotics_2024/)

## Demo

[![Demo Video](https://img.youtube.com/vi/xDCfFlaXbyM/0.jpg)](https://youtu.be/xDCfFlaXbyM)

## Acknowledgements

[Maze Runner 2D, by Al Sweigart al@inventwithpython.com Maze files are generated by mazemakerrec.py]
