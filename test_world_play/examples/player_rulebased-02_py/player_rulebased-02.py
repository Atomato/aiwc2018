#!/usr/bin/python3

# keunhyung 7/17
# new rule based player

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
            self.cur_my_posture = []
            self.cur_op_posture = []
            self.cur_ball = [0, 0]
            self.prev_ball = [0, 0]
            self.wheels = [0 for _ in range(10)]

            self.distances = [[i for i in range(5)], [i for i in range(5)]] # distances to the ball
            self.idxs = [[i for i in range(5)], [i for i in range(5)]]
            self.deadlock_cnt = 0
            self.avoid_deadlock_cnt = 0
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
        self.cur_my_posture = received_frame.coordinates[MY_TEAM]
        self.cur_op_posture =received_frame.coordinates[OP_TEAM]

    def sort_by_distance_to_ball(self, received_frame, team):
        # sort according to distant to the ball
        for i in range(self.number_of_robots):
            self.distances[team][i] = helper.distance(self.cur_ball[X], received_frame.coordinates[team][i][X], 
                self.cur_ball[Y], received_frame.coordinates[team][i][Y])
        self.idxs[MY_TEAM] = sorted(range(len(self.distances[team])), key=lambda k: self.distances[team][k])    

    def count_deadlock(self):
        d_ball = helper.distance(self.cur_ball[X], self.prev_ball[X], self.cur_ball[Y], self.prev_ball[Y]) # delta of ball
        #self.printConsole("boal delta: " + str(d_ball))
        if (abs(self.cur_ball[Y]) > 0.65) and (d_ball < 0.02):
            #self.printConsole("boal stop")
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

    def set_wheel_velocity(self, robot_id, left_wheel, right_wheel):
        multiplier = 1
        
        if(abs(left_wheel) > self.max_linear_velocity or abs(right_wheel) > self.max_linear_velocity):
            if (abs(left_wheel) > abs(right_wheel)):
                multiplier = self.max_linear_velocity / abs(left_wheel)
            else:
                multiplier = self.max_linear_velocity / abs(right_wheel)
        
        self.wheels[2*robot_id] = left_wheel*multiplier
        self.wheels[2*robot_id + 1] = right_wheel*multiplier

    def position(self, robot_id, x, y):
        damping = 0.35
        mult_lin = 3.5
        mult_ang = 0.4
        ka = 0
        sign = 1
        
        dx = x - self.cur_my_posture[robot_id][X]
        dy = y - self.cur_my_posture[robot_id][Y]
        d_e = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))
        desired_th = (math.pi/2) if (dx == 0 and dy == 0) else math.atan2(dy, dx)

        d_th = desired_th - self.cur_my_posture[robot_id][TH] 
        while(d_th > math.pi):
            d_th -= 2*math.pi
        while(d_th < -math.pi):
            d_th += 2*math.pi
            
        if (d_e > 1):
            ka = 17/90
        elif (d_e > 0.5):
            ka = 19/90
        elif (d_e > 0.3):
            ka = 21/90
        elif (d_e > 0.2):
            ka = 23/90
        else:
            ka = 25/90
            
        if (d_th > helper.degree2radian(95)):
            d_th -= math.pi
            sign = -1
        elif (d_th < helper.degree2radian(-95)):
            d_th += math.pi
            sign = -1
            
        if (abs(d_th) > helper.degree2radian(85)):
            self.set_wheel_velocity(robot_id, -mult_ang*d_th, mult_ang*d_th)
        else:
            if (d_e < 5 and abs(d_th) < helper.degree2radian(40)):
                ka = 0.1
            ka *= 4
            self.set_wheel_velocity(robot_id, 
                                    sign * (mult_lin * (1 / (1 + math.exp(-3*d_e)) - damping) - mult_ang * ka * d_th),
                                    sign * (mult_lin * (1 / (1 + math.exp(-3*d_e)) - damping) + mult_ang * ka * d_th))
            
    @inlineCallbacks
    def on_event(self, f):        

        @inlineCallbacks
        def set_wheel(self, robot_wheels):
            yield self.call(u'aiwc.set_speed', args.key, robot_wheels)
            return

        def goalie(self, robot_id):
            # Goalie just track the ball[Y] position at a fixed position on the X axis
            x = (-self.field[X]/2) + (self.robot_size/2) + 0.05
            y = max(min(self.cur_ball[Y], (self.goal[Y]/2 - self.robot_size/2)), -self.goal[Y]/2 + self.robot_size/2)
            self.position(robot_id, x, y)
            
        def defender(self, robot_id, idx, offset_y):
            ox = 0.1
            oy = 0.075
            min_x = (-self.field[X]/2) + (self.robot_size/2) + 0.05 
            
            # If ball is on offense
            if (self.cur_ball[X] > 0):
                # If ball is in the upper part of the field (y>0)
                if (self.cur_ball[Y] > 0):
                    self.position(robot_id, 
                                  (self.cur_ball[X]-self.field[X]/2)/2, 
                                  (min(self.cur_ball[Y],self.field[Y]/3))+offset_y)
                # If ball is in the lower part of the field (y<0)
                else:
                    self.position(robot_id, 
                                  (self.cur_ball[X]-self.field[X]/2)/2, 
                                  (max(self.cur_ball[Y],-self.field[Y]/3))+offset_y)
            # If ball is on defense
            else:
                # If robot is in front of the ball
                if (self.cur_my_posture[robot_id][X] > self.cur_ball[X] - ox):
                    # If this defender is the nearest defender from the ball
                    if (robot_id == idx):
                        self.position(robot_id, 
                                      (self.cur_ball[X]-ox), 
                                      ((self.cur_ball[Y]+oy) if (self.cur_my_posture[robot_id][Y]<0) else (self.cur_ball[Y]-oy)))
                    else:
                        self.position(robot_id, 
                                      (max(self.cur_ball[X]-0.03, min_x)), 
                                      ((self.cur_my_posture[robot_id][Y]+0.03) if (self.cur_my_posture[robot_id][Y]<0) else (self.cur_my_posture[robot_id][Y]-0.03)))
                # If robot is behind the ball
                else:
                    if (robot_id == idx):
                        self.position(robot_id, 
                                      self.cur_ball[X], 
                                      self.cur_ball[Y])                        
                    else:
                        self.position(robot_id, 
                                      (max(self.cur_ball[X]-0.03, min_x)), 
                                      ((self.cur_my_posture[robot_id][Y]+0.03) if (self.cur_my_posture[robot_id][Y]<0) else (self.cur_my_posture[robot_id][Y]-0.03)))

        def midfielder(self, robot_id):
            goal_dist = helper.distance(self.cur_my_posture[robot_id][X], self.field[X]/2, self.cur_my_posture[robot_id][Y], 0)            
            shoot_mul = 1
            dribble_dist = 0.426
            v = 5
            goal_to_ball_unit = helper.unit([self.field[X]/2 - self.cur_ball[X], - self.cur_ball[Y]])
            delta = [self.cur_ball[X] - self.cur_my_posture[robot_id][X], self.cur_ball[Y] - self.cur_my_posture[robot_id][Y]]

            if (self.distances[MY_TEAM][robot_id] < 0.5) and (delta[X] > 0):
                self.position(robot_id, self.cur_ball[X] + v*delta[X], self.cur_ball[Y] + v*delta[Y])                   
            else:
                self.position(robot_id, self.cur_ball[X] - dribble_dist*goal_to_ball_unit[X], self.cur_ball[Y] - dribble_dist*goal_to_ball_unit[Y])             
##############################################################################
        # new rule
        def chase(self, robot_id, x, y):
            self.position(robot_id, x, y)

        def avoid_deadlock(self):
            self.position(0, self.cur_ball[X], 0)
            self.position(1, self.cur_ball[X], 0)
            self.position(2, self.cur_ball[X], 0)
            self.position(3, self.cur_ball[X], 0)
            self.position(4, self.cur_ball[X], 0)

            if (self.distances[MY_TEAM][self.idxs[MY_TEAM][0]] > 0.2) or (self.avoid_deadlock_cnt > 20):
                # self.printConsole("                                                             return to the ball")
                midfielder(self, self.idxs[MY_TEAM][0])
                chase(self, self.idxs[MY_TEAM][1], self.cur_my_posture[self.idxs[MY_TEAM][0]][X], self.cur_my_posture[self.idxs[MY_TEAM][0]][Y])
                chase(self, self.idxs[MY_TEAM][2], self.cur_my_posture[self.idxs[MY_TEAM][1]][X], self.cur_my_posture[self.idxs[MY_TEAM][1]][Y])
                chase(self, self.idxs[MY_TEAM][3], self.cur_my_posture[self.idxs[MY_TEAM][2]][X], self.cur_my_posture[self.idxs[MY_TEAM][2]][Y])
                chase(self, self.idxs[MY_TEAM][4], self.cur_my_posture[self.idxs[MY_TEAM][3]][X], self.cur_my_posture[self.idxs[MY_TEAM][3]][Y])                
        
        def avoid_goal_foul(self):
            midfielder(self, self.idxs[MY_TEAM][0])
            chase(self, self.idxs[MY_TEAM][1], self.cur_my_posture[self.idxs[MY_TEAM][0]][X], self.cur_my_posture[self.idxs[MY_TEAM][0]][Y])
            self.position(self.idxs[MY_TEAM][2], 0, 0)
            self.position(self.idxs[MY_TEAM][3], 0, 0)
            self.position(self.idxs[MY_TEAM][4], 0, 0)

        def avoid_penalty_foul(self):
            midfielder(self, self.idxs[MY_TEAM][0])
            chase(self, self.idxs[MY_TEAM][1], self.cur_my_posture[self.idxs[MY_TEAM][0]][X], self.cur_my_posture[self.idxs[MY_TEAM][0]][Y])
            chase(self, self.idxs[MY_TEAM][2], self.cur_my_posture[self.idxs[MY_TEAM][1]][X], self.cur_my_posture[self.idxs[MY_TEAM][1]][Y])
            self.position(self.idxs[MY_TEAM][3], 0, 0)
            self.position(self.idxs[MY_TEAM][4], 0, 0)            
##############################################################################                
            
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
            self.sort_by_distance_to_ball(received_frame, MY_TEAM)
            self.count_deadlock()
            goal_area_count = self.count_goal_area()
            penalty_area_count = self.count_penalty_area()
        
            if goal_area_count > 2:
                # self.printConsole("                                                     goal foul!")
                avoid_goal_foul(self)
            elif penalty_area_count > 3:
                # self.printConsole("                                                     penalty foul!")
                avoid_penalty_foul(self)
            elif self.deadlock_cnt > 15:
                # self.printConsole("                        warning: deadlock")
                # self.printConsole("robot " + str(self.idxs[MY_TEAM][0]) + " distance: " + str(self.distances[MY_TEAM][self.idxs[MY_TEAM][0]]))
                avoid_deadlock(self)              
                self.avoid_deadlock_cnt += 1  
            else:
                midfielder(self, self.idxs[MY_TEAM][0])
                midfielder(self, self.idxs[MY_TEAM][1])
                # chase(self, self.idxs[MY_TEAM][1], self.cur_my_posture[self.idxs[MY_TEAM][0]][X], self.cur_my_posture[self.idxs[MY_TEAM][0]][Y])
                chase(self, self.idxs[MY_TEAM][2], self.cur_my_posture[self.idxs[MY_TEAM][1]][X], self.cur_my_posture[self.idxs[MY_TEAM][1]][Y])
                chase(self, self.idxs[MY_TEAM][3], self.cur_my_posture[self.idxs[MY_TEAM][2]][X], self.cur_my_posture[self.idxs[MY_TEAM][2]][Y])
                chase(self, self.idxs[MY_TEAM][4], self.cur_my_posture[self.idxs[MY_TEAM][3]][X], self.cur_my_posture[self.idxs[MY_TEAM][3]][Y])
##############################################################################
            self.prev_ball = self.cur_ball
            set_wheel(self, self.wheels)         
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
    
    # create a Wamp session object
    session = Component(ComponentConfig(ai_realm, {}))

    # initialize the msgpack serializer
    serializer = MsgPackSerializer()
    
    # use Wamp-over-rawsocket
    runner = ApplicationRunner(ai_sv, ai_realm, serializers=[serializer])
    
    runner.run(session, auto_reconnect=True)
