import numpy as np
import pystk
import matplotlib.pyplot as plt
from model.puck_detector import PuckDetector
from model.puck_is import PuckIs
import torch
from torchvision.transforms import functional as F
import torchvision
from PIL import Image
from torch import load
from os import path

GOAL_0 = np.array([0, 64.5])
GOAL_1 = np.array([0, -64.5])
PLAYER_LOC = np.array([200, 180])

FIRST = True
IM = None
BACKUP = False
PREV_DRAW = None
PREV_DRAW2 = None
LAST_X = None

def load_detector():
    r = PuckDetector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '../model/puck_det.th'), map_location='cpu'))
    return r

def load_is():
    r = PuckIs()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '../model/puck_is.th'), map_location='cpu'))
    return r

class HockeyPlayer:
    """
       Your ice hockey player. You may do whatever you want here. There are three rules:
        1. no calls to the pystk library (your code will not run on the tournament system if you do)
        2. There needs to be a deep network somewhere in the loop
        3. You code must run in 100 ms / frame on a standard desktop CPU (no for testing GPU)
        
        Try to minimize library dependencies, nothing that does not install through pip on linux.
    """
    
    """
       You may request to play with a different kart.
       Call `python3 -c "import pystk; pystk.init(pystk.GraphicsConfig.ld()); print(pystk.list_karts())"` to see all values.
    """
    kart = ""
    
    def __init__(self, player_id = 0):
        """
        Set up a soccer player.
        The player_id starts at 0 and increases by one for each player added. You can use the player id to figure out your team (player_id % 2), or assign different roles to different agents.
        """
        self.player_id = player_id
        self.kart = 'wilber'
        self.team = player_id % 2
        self.puck_detector = load_detector()
        self.puck_detector.eval()
        # self.puck_is = load_is()
        # self.puck_is.eval()
        self.resize = torchvision.transforms.Resize([150, 200])
      
    def act(self, image, player_info):
        global FIRST, IM, BACKUP, PREV_DRAW, PREV_DRAW2, LAST_X

        score_goal = None
        if self.team == 0:
            score_goal = GOAL_0
        else:
            score_goal = GOAL_1

        front = np.array(player_info.kart.front)[[0,2]]
        kart = np.array(player_info.kart.location)[[0,2]]

        # neural network gets location of puck
        
        device = torch.device('cpu')

        I = Image.fromarray(image)
        I = self.resize(I)
        I = F.to_tensor(I)
        I = I[None, :]
        I = I.to(device)

        puck_data = self.puck_detector(I)
        puck_data = puck_data.detach().numpy()[0]

        # puck_present = self.puck_is(I)
        # puck_present = puck_present.detach().numpy().item()

        puck_x = puck_data[0]
        puck_y = puck_data[1]

        if player_info.kart.id == 0:
            print('-------------------')
            print('puck_x before: ', puck_x)

        # player to goal
        v = front - score_goal
        v = 70 * (v / np.linalg.norm(v))

        new_puck_x = puck_x + v[0]
        
        if player_info.kart.id == 0:
            print('adj: ', v[0])
            print('puck_x after: ', new_puck_x)

        steer = ((new_puck_x - 200) / 400) * 20
        accel = 1
        brake = False
        drift = False

        if puck_y > 225 and puck_x > 150 and puck_x < 250:
            accel = 0
            brake = True
            steer = -0.6

        # if puck_present < 50:
        #     accel = 0
        #     brake = True
        #     if LAST_X is None:
        #         steer = -0.5
        #     elif LAST_X < 0:
        #         steer = 0.6
        #     elif LAST_X >= 0:
        #         steer = -0.6

        # visualize the controller in real time
        if player_info.kart.id == 0:
            ax1 = plt.subplot(111)
            if FIRST:
                IM = ax1.imshow(image)
                FIRST = False
            else:
                IM.set_data(image)

            if PREV_DRAW is not None:
                PREV_DRAW.remove()
                PREV_DRAW2.remove()

            test = plt.Circle(PLAYER_LOC, 10, ec='b', fill=False, lw=1.5)
            ax1.add_artist(test)

            PREV_DRAW = plt.Circle(puck_data, 10, ec='g', fill=False, lw=1.5)
            PREV_DRAW2 = plt.Circle((new_puck_x, puck_y), 10, ec='r', fill=False, lw=1.5)
            ax1.add_artist(PREV_DRAW)
            ax1.add_artist(PREV_DRAW2)

            # if puck_present < 50:
            #     print('PUCK out of sight')

            plt.pause(0.001)

        # if puck_present > 50:
        #     LAST_X = puck_x

        action = {
            'steer': steer,
            'acceleration': accel,
            'brake': brake,
            'drift': drift,
            'nitro': False, 
            'rescue': False}

        return action



class GoaliePlayer:
    
    kart = ""
    
    def __init__(self, player_id = 0):
        """
        Set up a soccer player.
        The player_id starts at 0 and increases by one for each player added. You can use the player id to figure out your team (player_id % 2), or assign different roles to different agents.
        """
        self.player_id = player_id
        self.kart = 'tux'
        self.team = player_id % 2
        self.puck_detector = load_model()
        self.puck_detector.eval()
        #self.goalie_up_direction = True
        #self.goalie_postion = 0
      
    def act(self, image, player_info):
        global FIRST, IM, BACKUP
        
        other_goal = None
        team_goal = None
        
        print(self.team)
        if self.team == 0:
            team_goal = GOAL_1
            other_goal = GOAL_0
        else:
            team_goal = GOAL_0
            other_goal = GOAL_1

        front = np.array(player_info.kart.front)[[0,2]]
        location = np.array(player_info.kart.location)[[0,2]]

        device = torch.device('cpu')
        I = F.to_tensor(image)
        I = I.to(device)
        puck_loc = self.puck_detector(I)
        puck_loc = puck_loc.detach().numpy()[0]

        #too_far = False
        
        puck_x = puck_loc[0]
        puck_y = puck_loc[1]

        if(puck_y > 200 and puck_x >= 150 and puck_x <= 250):
            u = location - front
            u = u / np.linalg.norm(u)

            v = location - team_goal
            v = v / np.linalg.norm(v)

            accel = 0.1
            brake = False
            drift = False

            dist_to_goal = np.linalg.norm(v)
            if(dist_to_goal < 1):
              v = location - other_goal
     
            theta = np.arccos(np.dot(u, v))
            signed_theta = -np.sign(np.cross(u, v)) * theta
            steer = 20 * signed_theta
            

        else:
          u = location - puck_loc
          u = u / np.linalg.norm(u)
          dist_to_puck = np.linalg.norm(u)
          if(dist_to_puck < 20):
            v = location - front
            v = v / np.linalg.norm(v)

            theta = np.arccos(np.dot(u, v))
            signed_theta = -np.sign(np.cross(u, v)) * theta


            steer = 20 * signed_theta
            accel = 0.1
            brake = False
            drift = False
          else:
            steer = 0
            accel = 0
            brake = False
            drift = False


        if np.degrees(theta) > 60 and np.degrees(theta) < 90:
                drift = True

            if np.degrees(theta) > 90 and not BACKUP:
                BACKUP = True

            if BACKUP:
                if np.degrees(theta) > 30:
                    accel = 0
                    brake = True
                    steer = -steer
                else:
                    BACKUP = False
          
            

        # visualize the controller in real time
        if player_info.kart.id == 0:
            ax1 = plt.subplot(111)
            if FIRST:
                IM = ax1.imshow(image)
                FIRST = False
            else:
                IM.set_data(image)

            #ax1.add_artist(plt.Circle(puck_loc, 10, ec='g', fill=False, lw=1.5))
            #print('loc: ' + str(location))
            #print('team goal loc: ' + str(goal_point))
            plt.pause(0.001)

        action = {
            'steer': steer,
            'acceleration': accel,
            'brake': brake,
            'drift': drift,
            'nitro': False, 
            'rescue': False}

        return action