#!/usr/bin/python3

from __future__ import print_function

from twisted.internet import reactor
from twisted.internet.defer import inlineCallbacks

from autobahn.wamp.serializer import MsgPackSerializer
from autobahn.wamp.types import ComponentConfig
from autobahn.twisted.wamp import ApplicationSession, ApplicationRunner

import argparse
import random
import math
import os
import sys

import base64
import numpy as np

import helper

import tensorflow as tf

from args import Argument
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
from discrete import Discrete

#reset_reason
NONE = 0
GAME_START = 1
SCORE_MYTEAM = 2
SCORE_OPPONENT = 3
GAME_END = 4
DEADLOCK = 5

#coordinates
MY_TEAM = 0
OP_TEAM = 1
BALL = 2
X = 0
Y = 1
TH = 2
ACTIVE = 3
TOUCH = 4

class Frame(object):
    def __init__(self):
        self.time = None
        self.score = None
        self.reset_reason = None
        self.coordinates = None

class Component(ApplicationSession):

    def __init__(self, config):
        ApplicationSession.__init__(self, config)

    def printConsole(self, message):
        print(message)
        sys.__stdout__.flush()

    def onConnect(self):
        self.join(self.config.realm)

    @inlineCallbacks
    def onJoin(self, details):

##############################################################################
        def init_variables(self, info):
            # Here you have the information of the game (virtual init() in random_walk.cpp)
            # List: game_time, goal, number_of_robots, penalty_area, codewords,
            #       robot_height, robot_radius, max_linear_velocity, field, team_info,
            #       {rating, name}, axle_length, resolution, ball_radius
            # self.game_time = info['game_time']
            self.field = info['field']
            self.robot_size = 2*info['robot_radius']
            self.goal = info['goal']
            self.max_linear_velocity = info['max_linear_velocity']
            self.number_of_robots = info['number_of_robots']
            self.end_of_frame = False
            self.cur_my_posture = []
            self.cur_op_posture = []
            self.cur_ball = []
            self.pre_ball = [0, 0]

            self.state_dim = 2 # relative ball
            self.history_size = 2 # frame history size
            self.action_dim = 2 # 2                    
            
            self.arglist = Argument()
            self.obs_shape_n = [(self.state_dim * self.history_size,) for _ in range(1)] # state dimenstion
            self.action_space = [Discrete(self.action_dim * 2 + 1) for _ in range(1)]
            self.trainers = self.get_trainers(1, self.obs_shape_n, self.action_space, self.arglist)

            U.initialize()
            
            # Load previous results, if necessary
            if self.arglist.load_dir == "":
                self.arglist.load_dir = self.arglist.save_dir
            if self.arglist.display or self.arglist.restore or self.arglist.benchmark:
                print('Loading previous state...')
                U.load_state(self.arglist.load_dir)

            self.obs_n = [np.zeros([self.state_dim * self.history_size]) for _ in range(self.number_of_robots)] # histories
            self.wheels = np.zeros(self.number_of_robots*2)
            self.action_n = [np.zeros(self.action_dim * 2 + 1) for _ in range(self.number_of_robots)] # not np.zeros(2)

            self.distances = [[i for i in range(5)], [i for i in range(5)]] # distances to the ball
            self.idxs = [[i for i in range(5)], [i for i in range(5)]]
            self.shoot_plan = [0 for _ in range(self.number_of_robots)]
            self.deadlock_cnt = 0
            self.avoid_deadlock_cnt = 0
            self.global_step = 0
            return
##############################################################################
            
        try:
            info = yield self.call(u'aiwc.get_info', args.key)
        except Exception as e:
            self.printConsole("Error: {}".format(e))
        else:
            try:
                self.sub = yield self.subscribe(self.on_event, args.key)
            except Exception as e2:
                self.printConsole("Error: {}".format(e2))
               
        init_variables(self, info)
        
        try:
            yield self.call(u'aiwc.ready', args.key)
        except Exception as e:
            self.printConsole("Error: {}".format(e))
        else:
            self.printConsole("I am ready for the game!")

    def mlp_model(self, input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
        # This model takes as input an observation and returns values of all actions
        with tf.variable_scope(scope, reuse=reuse):
            out = input
            out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
            return out

    def get_trainers(self, num_agent, obs_shape_n, action_space, arglist):
        trainers = []
        model = self.mlp_model
        trainer = MADDPGAgentTrainer

        for i in range(num_agent):
            trainers.append(trainer(
                "agent_%d" % i, model, obs_shape_n, action_space, i, arglist,
                local_q_func=(arglist.good_policy=='ddpg')))
        return trainers
##############################################################################    
    def get_coord(self, received_frame):
        self.cur_ball = received_frame.coordinates[BALL]
        self.cur_my_posture = received_frame.coordinates[MY_TEAM]
        self.cur_op_posture =received_frame.coordinates[OP_TEAM]   

    def pre_processing(self, i, x, y):
        relative_pos = helper.rot_transform(self.cur_my_posture[i][X], 
            self.cur_my_posture[i][Y], -self.cur_my_posture[i][TH], x, y)

        self.obs_n[i] = np.append(relative_pos, relative_pos - self.obs_n[i][:-self.state_dim])

    def sort_by_distance_to_ball(self, received_frame, team):
        # sort according to distant to the ball
        for i in range(self.number_of_robots):
            self.distances[team][i] = helper.distance(self.cur_ball[X], received_frame.coordinates[team][i][X], 
                self.cur_ball[Y], received_frame.coordinates[team][i][Y])
        self.idxs[MY_TEAM] = sorted(range(len(self.distances[team])), key=lambda k: self.distances[team][k])        

    def count_deadlock(self):
        d_ball = helper.distance(self.cur_ball[X], self.pre_ball[X], self.cur_ball[Y], self.pre_ball[Y]) # delta of ball
        #self.printConsole("boal delta: " + str(d_ball))
        if (abs(self.cur_ball[Y]) > 0.65) and (d_ball < 0.015):
            self.printConsole("                         boal stop " + str(self.deadlock_cnt))
            self.deadlock_cnt += 1
        else:
            self.deadlock_cnt = 0
            self.avoid_deadlock_cnt = 0

    def count_goal_area(self):
        count = 0
        for i in range(self.number_of_robots):
            if (abs(self.cur_my_posture[i][X]) > 1.6) and (abs(self.cur_my_posture[i][Y]) < 0.43):
                count += 1
        return count

    def count_penalty_area(self):
        count = 0
        for i in range(self.number_of_robots):
            if (abs(self.cur_my_posture[i][X]) > 1.3) and (abs(self.cur_my_posture[i][Y]) < 0.7):
                count += 1
        return count            

    @inlineCallbacks
    def on_event(self, f):        

        @inlineCallbacks
        def set_wheel(self, robot_wheels):
            yield self.call(u'aiwc.set_speed', args.key, robot_wheels)
            return

        def avoid_goal_foul(self):
            x = -1.8 if self.cur_ball[X] < 0 else 1.8
            shooter(self, self.idxs[MY_TEAM][0])
            shooter(self, self.idxs[MY_TEAM][1])
            self.pre_processing(self.idxs[MY_TEAM][2], x, 0)
            self.pre_processing(self.idxs[MY_TEAM][3], x, 0)
            self.pre_processing(self.idxs[MY_TEAM][4], x, 0)

        def avoid_penalty_foul(self):
            x = -1.3 if self.cur_ball[X] < 0 else 1.3
            shooter(self, self.idxs[MY_TEAM][0])
            shooter(self, self.idxs[MY_TEAM][1])
            self.pre_processing(self.idxs[MY_TEAM][2], self.cur_my_posture[self.idxs[MY_TEAM][1]][X], self.cur_my_posture[self.idxs[MY_TEAM][1]][Y])
            self.pre_processing(self.idxs[MY_TEAM][3], x, 0)
            self.pre_processing(self.idxs[MY_TEAM][4], x, 0)

        def avoid_deadlock(self):
            ox = []
            oy = []
            for i in range(5):
                ox.append(0.3 if self.cur_my_posture[i][X] > self.cur_ball[X] else -0.3)
                oy.append(0.3 if self.cur_my_posture[i][Y] > self.cur_ball[Y] else -0.3)

            if (self.distances[MY_TEAM][self.idxs[MY_TEAM][0]] > 0.2) or (self.avoid_deadlock_cnt > 30):
                self.printConsole("                                                             return to the ball")
                shooter(self, self.idxs[MY_TEAM][0])
                shooter(self, self.idxs[MY_TEAM][1])
                self.pre_processing(self.idxs[MY_TEAM][2], self.cur_my_posture[self.idxs[MY_TEAM][1]][X], self.cur_my_posture[self.idxs[MY_TEAM][1]][Y])
                self.pre_processing(self.idxs[MY_TEAM][3], self.cur_my_posture[self.idxs[MY_TEAM][2]][X], self.cur_my_posture[self.idxs[MY_TEAM][2]][Y])
                self.pre_processing(self.idxs[MY_TEAM][4], self.cur_my_posture[self.idxs[MY_TEAM][3]][X], self.cur_my_posture[self.idxs[MY_TEAM][3]][Y])
            else:
                self.pre_processing(0, self.cur_my_posture[0][X]+ox[0], self.cur_my_posture[0][Y]+oy[0])
                self.pre_processing(1, self.cur_my_posture[1][X]+ox[1], self.cur_my_posture[1][Y]+oy[1])
                self.pre_processing(2, self.cur_my_posture[2][X]+ox[2], self.cur_my_posture[2][Y]+oy[2])
                self.pre_processing(3, self.cur_my_posture[3][X]+ox[3], self.cur_my_posture[3][Y]+oy[3])
                self.pre_processing(4, self.cur_my_posture[4][X]+ox[4], self.cur_my_posture[4][Y]+oy[4])

        def shooter(self, i):
            ball = self.cur_ball

            if (self.cur_my_posture[i][X] > self.cur_ball[X]) and (self.cur_my_posture[i][X] > -1.7):
                self.shoot_plan[i] = 1
                ball[X] -= 0.3
                if self.cur_my_posture[i][Y] > self.cur_ball[Y]:
                    ball[Y] += 0.15
                else:
                    ball[Y] -= 0.15
                self.printConsole('         shoot plane ' + str(i) + ': ' + str(self.shoot_plan[i]))
            elif self.shoot_plan[i] is 1:
                if self.cur_my_posture[i][X] < self.cur_ball[X] -0.2:
                    self.shoot_plan[i] = 0
                    self.printConsole('         shoot plane ' + str(i) + ': ' + str(self.shoot_plan[i]))
                else:
                    ball[X] -= 0.3
                    if self.cur_my_posture[i][Y] > self.cur_ball[Y]:
                        ball[Y] += 0.15
                    else:
                        ball[Y] -= 0.15
                    self.printConsole('         shoot plane ' + str(i) + ': ' + str(self.shoot_plan[i]))
            else:
                self.printConsole('         shoot plane ' + str(i) + ': ' + str(self.shoot_plan[i]))


            self.pre_processing(i, ball[X], ball[Y])
            
        # initiate empty frame
        received_frame = Frame()

        if 'time' in f:
            received_frame.time = f['time']
        if 'score' in f:
            received_frame.score = f['score']
        if 'reset_reason' in f:
            received_frame.reset_reason = f['reset_reason']
        if 'coordinates' in f:
            received_frame.coordinates = f['coordinates']            
        if 'EOF' in f:
            self.end_of_frame = f['EOF']
        
        #self.printConsole(received_frame.time)
        #self.printConsole(received_frame.score)
        #self.printConsole(received_frame.reset_reason)
        #self.printConsole(self.end_of_frame)
##############################################################################
        if (self.end_of_frame):
            
            # How to get the robot and ball coordinates: (ROBOT_ID can be 0,1,2,3,4)
            #self.printConsole(received_frame.coordinates[MY_TEAM][ROBOT_ID][X])            
            #self.printConsole(received_frame.coordinates[MY_TEAM][ROBOT_ID][Y])
            #self.printConsole(received_frame.coordinates[MY_TEAM][ROBOT_ID][TH])
            #self.printConsole(received_frame.coordinates[MY_TEAM][ROBOT_ID][ACTIVE])
            #self.printConsole(received_frame.coordinates[MY_TEAM][ROBOT_ID][TOUCH])
            #self.printConsole(received_frame.coordinates[OP_TEAM][ROBOT_ID][X])
            #self.printConsole(received_frame.coordinates[OP_TEAM][ROBOT_ID][Y])
            #self.printConsole(received_frame.coordinates[OP_TEAM][ROBOT_ID][TH])
            #self.printConsole(received_frame.coordinates[OP_TEAM][ROBOT_ID][ACTIVE])
            #self.printConsole(received_frame.coordinates[OP_TEAM][ROBOT_ID][TOUCH])
            #self.printConsole(received_frame.coordinates[BALL][X])
            #self.printConsole(received_frame.coordinates[BALL][Y])
                                               
            self.get_coord(received_frame)
##############################################################################
            self.printConsole('time step: ' + str(self.global_step))
            self.sort_by_distance_to_ball(received_frame, MY_TEAM)
            self.count_deadlock()
            goal_area_count = self.count_goal_area()
            penalty_area_count = self.count_penalty_area()

            if goal_area_count > 2:
                self.printConsole("                                                     goal foul!")
                avoid_goal_foul(self)
            elif penalty_area_count > 3:
                self.printConsole("                                                     penalty foul!")
                avoid_penalty_foul(self)
            elif self.deadlock_cnt > 10:
                self.printConsole("                        warning: deadlock")
                avoid_deadlock(self)              
                self.avoid_deadlock_cnt += 1                
            else:
                shooter(self, self.idxs[MY_TEAM][0])
                shooter(self, self.idxs[MY_TEAM][1])
                self.pre_processing(self.idxs[MY_TEAM][2], self.cur_my_posture[self.idxs[MY_TEAM][1]][X], self.cur_my_posture[self.idxs[MY_TEAM][1]][Y])
                self.pre_processing(self.idxs[MY_TEAM][3], self.cur_my_posture[self.idxs[MY_TEAM][2]][X], self.cur_my_posture[self.idxs[MY_TEAM][2]][Y])
                self.pre_processing(self.idxs[MY_TEAM][4], self.cur_my_posture[self.idxs[MY_TEAM][3]][X], self.cur_my_posture[self.idxs[MY_TEAM][3]][Y])                


            # get action
            self.action_n = [self.trainers[0].action(obs) for obs in self.obs_n]

            for i in range(self.number_of_robots):
                self.wheels[2*i] = self.max_linear_velocity * (self.action_n[i][1]-self.action_n[i][2]+self.action_n[i][3]-self.action_n[i][4])
                self.wheels[2*i + 1] = self.max_linear_velocity * (self.action_n[i][1]-self.action_n[i][2]-self.action_n[i][3]+self.action_n[i][4])

            self.pre_ball = self.cur_ball
            self.global_step += 1
            set_wheel(self, self.wheels.tolist())           
##############################################################################
            if(received_frame.reset_reason == GAME_END):
                #(virtual finish() in random_walk.cpp)
                #save your data
                with open(args.datapath + '/result.txt', 'w') as output:
                    #output.write('yourvariables')
                    output.close()
                #unsubscribe; reset or leave  
                yield self.sub.unsubscribe()
                try:
                    yield self.leave()
                except Exception as e:
                    self.printConsole("Error: {}".format(e))

            self.end_of_frame = False
##############################################################################

    def onDisconnect(self):
        if reactor.running:
            reactor.stop()

if __name__ == '__main__':
    
    try:
        unicode
    except NameError:
        # Define 'unicode' for Python 3
        def unicode(s, *_):
            return s

    def to_unicode(s):
        return unicode(s, "utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("server_ip", type=to_unicode)
    parser.add_argument("port", type=to_unicode)
    parser.add_argument("realm", type=to_unicode)
    parser.add_argument("key", type=to_unicode)
    parser.add_argument("datapath", type=to_unicode)
    
    args = parser.parse_args()
    
    ai_sv = "rs://" + args.server_ip + ":" + args.port
    ai_realm = args.realm
    
    with U.single_threaded_session():
        # create a Wamp session object
        session = Component(ComponentConfig(ai_realm, {}))

        # initialize the msgpack serializer
        serializer = MsgPackSerializer()
        
        # use Wamp-over-rawsocket
        runner = ApplicationRunner(ai_sv, ai_realm, serializers=[serializer])
        
        runner.run(session, auto_reconnect=True)
