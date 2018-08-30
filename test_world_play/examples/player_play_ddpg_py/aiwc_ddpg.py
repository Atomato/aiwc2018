import random
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Input, Concatenate, Lambda
from keras import backend as K

import sys

# tensoflow
# DDPG Agent for the soccer robot 
# continous action control
class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_max):
        # load model if True
        self.load_model = True

        tf.reset_default_graph()
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))        

        # information of state and action
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_max = float(action_max)
        self.action_min = -float(action_max)

        # hyper parameters
        self.h_critic = 16
        self.h_actor = 16

        self.state_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim])

        with tf.variable_scope('actor'):
            self.action = self.generate_actor_network(self.state_ph, True)

        self.sess.run(tf.global_variables_initializer())

        self.load_file = "./save_model/chase_shoot/tensorflow_ddpg-1"
        self.saver = tf.train.Saver()
        if self.load_model:
            self.saver.restore(self.sess, self.load_file)

    def generate_actor_network(self, state, trainable):
        hidden1 = tf.layers.dense(state, self.h_actor, activation=tf.nn.relu, trainable=trainable)
        hidden2 = tf.layers.dense(hidden1, self.h_actor, activation=tf.nn.relu, trainable=trainable)
        hidden3 = tf.layers.dense(hidden2, self.h_actor, activation=tf.nn.relu, trainable=trainable)

        non_scaled_action = tf.layers.dense(hidden3, self.action_dim, activation=tf.nn.sigmoid, trainable=trainable)
        action = non_scaled_action * (self.action_max - self.action_min) + self.action_min

        return action

    def get_action(self, obs):
        action = self.sess.run(self.action, feed_dict = {self.state_ph: obs[None]})[0]
        return action

    def ou_noise(self, x):
        return x + self.theta * (self.mu-x) + self.sigma * np.random.randn(self.action_dim)

    def printConsole(self, message):
        print(message)
        sys.__stdout__.flush()

