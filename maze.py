import sys, os
class MazeClass():
    def __init__(self, file):
        
        self.file = file
        # Maze file constants:
        self.WALL = '#'              # when running the reset function in simple drive when self.it read the wall place wall
        self.EMPTY = ' '             
        self.START = 'S'             # place robot in this location
        self.EXIT = 'E'              # place goal in this location
        
    def readMazeFile(self):
        # Load the maze from a file:
        mazeFile = open(self.file)
        maze = {}
        lines = mazeFile.readlines()
        playerx = None
        playery = None
        exitx = None
        exity = None
        y = 0
        for line in lines:
            WIDTH = len(line.rstrip())
            for x, character in enumerate(line.rstrip()):
                assert character in (self.WALL, self.EMPTY, self.START, self.EXIT), 'Invalid character at column {}, line {}'.format(x + 1, y + 1)
                if character in (self.WALL, self.EMPTY):
                    maze[(x, y)] = character
                elif character == self.START:
                    playerx, playery = x, y
                    maze[(x, y)] = self.START
                elif character == self.EXIT:
                    exitx, exity = x, y
                    maze[(x, y)] = self.EXIT
            y += 1
        return maze