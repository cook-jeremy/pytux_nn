import numpy as np
import pystk
import matplotlib.pyplot as plt
from model.model import PuckDetector
import torch
from torchvision.transforms import functional as F
import torchvision
from PIL import Image

GOAL_0 = np.array([0, 64.5])
GOAL_1 = np.array([0, -64.5])

FIRST = True
IM = None
BACKUP = False
PREV_DRAW = None

def load_model():
    from torch import load
    from os import path
    r = PuckDetector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '../model/puck.th'), map_location='cpu'))
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
        self.kart = 'tux'
        self.team = player_id % 2
        self.puck_detector = load_model()
        self.puck_detector.eval()
        self.resize = torchvision.transforms.Resize([128, 128])
      
    def act(self, image, player_info):
        global FIRST, IM, BACKUP, PREV_DRAW

        score_goal = None
        print(self.team)
        if self.team == 0:
            score_goal = GOAL_0
        else:
            score_goal = GOAL_1

        front = np.array(player_info.kart.front)[[0,2]]
        location = np.array(player_info.kart.location)[[0,2]]

        # neural network gets location of puck
        
        device = torch.device('cpu')

        I = Image.fromarray(image)
        I = self.resize(I)
        I = F.to_tensor(I)
        I = I[None, :]
        I = I.to(device)
        puck_loc = self.puck_detector(I)
        puck_loc = puck_loc.detach().numpy()[0]

        u = front - location
        u = u / np.linalg.norm(u)

        v = puck_loc - location
        v = v / np.linalg.norm(v)

        theta = np.arccos(np.dot(u, v))
        signed_theta = -np.sign(np.cross(u, v)) * theta

        steer = 20 * signed_theta
        accel = 0.5
        accel = 0.1
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

            if PREV_DRAW is not None:
                PREV_DRAW.remove()

            PREV_DRAW = plt.Circle(puck_loc, 10, ec='g', fill=False, lw=1.5)
            ax1.add_artist(PREV_DRAW)
            #print('loc: ' + str(location))
            #print('puck loc: ' + str(puck_loc))
            plt.pause(0.001)

        action = {
            'steer': steer,
            'acceleration': accel,
            'brake': brake,
            'drift': drift,
            'nitro': False, 
            'rescue': False}

        return action