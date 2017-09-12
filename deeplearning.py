import tensorflow as tf
import cv2
import sys

sys.path.append("game/")


import wrapped_flappy_bird as game
import random
import numpy as np

from collections import deque



GAME             = 'superbird'
ACTIONS          = 2
GAMMA            = 0.99
OBSERVE          = 100000.
EXPLORE          = 2000000.
FINAL_EPSILON    = 0.0001
INITIAL_EPSILON  = 0.0001
REPLAY_MEMORY    = 50000
BATCH            = 32
FRAME_PER_ACTION = 1


def weight_variable(shape);
def bias_variable(shape);
def conv2d(x, W, stride);
def max_pool_2x2(x);
def createNetwork();
def trainNetwork(s, readout, h_fc1, sess);
def playGame();
def main():
    playGame()



if __name__ == '__main__':
    main()
