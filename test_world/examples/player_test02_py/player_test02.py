#!/usr/bin/python3

# keunhyung 7/13
# keras dqn refer to atari breakout 

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

from aiwc_dqn import DQNAgent

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
    """
    AI Base + Skeleton
    """ 

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
            self.cur_posture = []
            self.cur_ball = []
            self.prev_ball = []
            self.idx = 0
            self.wheels = [0 for _ in range(10)]

            self.state_size = 5 # the number of possible states
            self.action_size = 11 # the number of possible actions
            self.agent = DQNAgent(self.state_size, self.action_size)       
            self.global_step = 0 # iteration step             
            self.save_every_steps = 100 # save the model

            self.step = 0 # statistic step 
            self.stats_steps = 100 # for tensorboard
            self.reward_sum = 0
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

    def get_coord(self, received_frame):
        self.cur_ball = received_frame.coordinates[BALL]
        self.cur_posture = received_frame.coordinates[MY_TEAM]
            
    @inlineCallbacks
    def on_event(self, f):        

        @inlineCallbacks
        def set_wheel(self, robot_wheels):
            yield self.call(u'aiwc.set_speed', args.key, robot_wheels)
            return

        def set_action(robot_id, action_number):
            if action_number == 0:
                self.wheels[2*robot_id] = 0.75
                self.wheels[2*robot_id + 1] = 0.75
                # Go Forward with fixed velocity
            elif action_number == 1:
                self.wheels[2*robot_id] = 0.75
                self.wheels[2*robot_id + 1] = 0.5
                # Turn
            elif action_number == 2:
                self.wheels[2*robot_id] = 0.75
                self.wheels[2*robot_id + 1] = 0.25
                # Turn
            elif action_number == 3:
                self.wheels[2*robot_id] = 0.75
                self.wheels[2*robot_id + 1] = 0
                # Turn
            elif action_number == 4:
                self.wheels[2*robot_id] = 0.5
                self.wheels[2*robot_id + 1] = 75
                # Turn
            elif action_number == 5:
                self.wheels[2*robot_id] = 0.25
                self.wheels[2*robot_id + 1] = 0.75
                # Turn
            elif action_number == 6:
                self.wheels[2*robot_id] = 0
                self.wheels[2*robot_id + 1] = 0.75
                # Turn
            elif action_number == 7:
                self.wheels[2*robot_id] = -0.75
                self.wheels[2*robot_id + 1] = -0.75
                # Go Backward with fixed velocity
            elif action_number == 8:
                self.wheels[2*robot_id] = -0.1
                self.wheels[2*robot_id + 1] = 0.1
                # Spin
            elif action_number == 9:
                self.wheels[2*robot_id] = 0.1
                self.wheels[2*robot_id + 1] = -0.1
                # Spin
            elif action_number == 10:
                self.wheels[2*robot_id] = 0
                self.wheels[2*robot_id + 1] = 0
                # Do not move

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
                        
##############################################################################
            # next state
            self.get_coord(received_frame)
            next_state = [round(received_frame.coordinates[MY_TEAM][0][X]/2.05, 2), round(received_frame.coordinates[MY_TEAM][0][Y]/1.35, 2), round(received_frame.coordinates[MY_TEAM][0][TH]/(2*math.pi), 2),
                    round(received_frame.coordinates[BALL][X]/2.05, 2), round(received_frame.coordinates[BALL][Y]/1.35, 2)]
            next_state = np.reshape([next_state], (1, self.state_size, 1))
            next_history = np.append(next_state, self.agent.history[:, :, :3], axis=2)
            # Reward
            reward = 3*math.exp(-10*(helper.distance(self.cur_posture[0][X], self.cur_ball[X], self.cur_posture[0][Y], self.cur_ball[Y])/20))
            if self.cur_posture[0][TOUCH]:
                reward += 100
                self.printConsole("ball touch, reward is " + str(reward))
            # Reset
            reset = (received_frame.reset_reason == SCORE_MYTEAM) or (received_frame.reset_reason == SCORE_OPPONENT) or (received_frame.reset_reason == DEADLOCK)
##############################################################################            
            self.agent.avg_q_max += np.amax(
                self.agent.model.predict(np.reshape(self.agent.history, (1, -1)))[0])
            #self.printConsole("history: " + str(next_history))
##############################################################################            
            # save the sample <s, a, r, s'> to the replay memory
            self.agent.replay_memory(self.agent.history, self.agent.action, reward, next_history, reset)
            # every time step do the training
            if len(self.agent.memory) >= self.agent.train_start:
                self.agent.train_replay()

            # save the history and get action
            self.agent.history = next_history
            self.agent.action = self.agent.get_action(np.reshape(self.agent.history, (1, -1)))
            #self.printConsole("agent.action: " + str(action))
            # Set robot wheels
            set_action(0, self.agent.action)
            set_wheel(self, self.wheels)

            # go one step
            self.global_step += 1
            self.step += 1
            self.reward_sum += reward            
##############################################################################
            if self.global_step % 50 == 0:
                self.printConsole("step: " + str(self.global_step) + ", Epsilon: " + str(self.agent.epsilon) + ", reward: " + str(reward))

            if self.global_step % self.agent.update_target_rate == 0:
                # every reset update the target model to be same with model
                self.agent.update_target_model()
                self.printConsole("update target")

            if self.global_step % self.save_every_steps == 0: # save the model
                self.agent.save_model("./save_model/aiwc_dqn.h5")
                self.printConsole("Saved model")

            if self.global_step % self.stats_steps == 0: # plot the statics
                stats = [self.reward_sum, self.agent.avg_q_max / float(self.step), self.step,
                             self.agent.avg_loss / float(self.step)]
                for i in range(len(stats)):
                    self.agent.sess.run(self.agent.update_ops[i], feed_dict={
                        self.agent.summary_placeholders[i]: float(stats[i])
                    })
                summary_str = self.agent.sess.run(self.agent.summary_op)
                self.agent.summary_writer.add_summary(summary_str, self.global_step / self.stats_steps)

                self.printConsole("average reward: " + str(self.reward_sum / float(self.step)) + 
                    ", average_q: " + str(self.agent.avg_q_max / float(self.step)) + 
                    ", average loss: " + str(self.agent.avg_loss / float(self.step)))
                self.reward_sum, self.agent.avg_q_max, self.agent.avg_loss = 0, 0, 0
                self.step = 0
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
    
    # create a Wamp session object
    session = Component(ComponentConfig(ai_realm, {}))

    # initialize the msgpack serializer
    serializer = MsgPackSerializer()
    
    # use Wamp-over-rawsocket
    runner = ApplicationRunner(ai_sv, ai_realm, serializers=[serializer])
    
    runner.run(session, auto_reconnect=True)
