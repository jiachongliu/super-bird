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


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    """
    truncated_normal(shape,mean,stddev) -- 这个函数是用来产生正太分布,均值为mean,但在这个函数中没有使用，缺省值
    shape(生成张量的纬度) stddev(标准差)

    """
    return tf.Variable(intial)   #Variable(变量)  设定变量

def bias_variable(shape):
    inital = tf.constant(0.01, shape = shape)
    """
    constant(value, dtype=None, shape=None, name='Const', verify_shape=False)
    上面这个函数是创建一个常数张量
    value -- 一个类型为dtype常量(常量列表)
    dtype -- 指定生成的张量的类型
    shape -- 可选参数, 指定生成的张量的纬度
    name  -- 可选参数, 指定生成的张量的名字
    verify_shape -- 可选参数，布尔类型  是否启用验证value的形状

    """
    return tf.Variable(inital)

def conv2d(x, W, stride):

    """
    tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
    上面这个函数是通过输入规定的四维input和filter计算二维卷积

    input: 一个张量。必须为这些类型之一: half, float32, float64
    filter: 一个张量。必须要和input的类型一致
    strides： 一个整数列表。长度为4的一维矩阵。input的每一个维度的滑动窗格的步骤。必须与使用data_format指定
的维度具有相同的顺序。
    padding: 字符串类型，可选值为："SAME", "VALID"／
    use_cudnn_on_gpu: 一个可选的布尔参数.默认为True
    data_format: 一个可选的字符串参数。可选值为："NHWC", "NHWC"。默认为"NHWC"
    name: 操作的名称

    """
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1],
            padding = "SAME")
def createNetwork():
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([512])


    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    s = tf.placeholder("float", [None, 80, 80, 4])

    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_con1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    readout = tf.matmul(h_fc1, W_fc2) + b_fc2
    
    return s, readout, h_fc1

def trainNetwork(s, readout, h_fc1, sess);
def playGame();
def main():
    playGame()



if __name__ == '__main__':
    main()
