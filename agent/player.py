import numpy as np
import pystk
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import functional as F
import torchvision
from PIL import Image
from os import path
import random

GOAL_0 = np.array([0, 64.5])
GOAL_1 = np.array([0, -64.5])
PLAYER_LOC = np.array([200, 180])

def load_detector():
    from model.puck_detector import PuckDetector
    from torch import load
    from os import path
    r = PuckDetector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '../model/puck_det.th'), map_location='cpu'))
    return r

# def load_vec():
#     from model.vec_detector import VecDetector
#     from torch import load
#     from os import path
#     r = VecDetector()
#     r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '../model/puck_vec.th'), map_location='cpu'))
#     return r

#     calculate the player to puck vector
#     p_info = []
#     p_info.extend(kart_xy)
#     p_info.extend(kart_to_front)
#     p_info.extend(screen_puck_xy)
#     p_ten = torch.Tensor(p_info)
#     kart_to_puck = self.vec_detector(p_ten).detach().numpy()
#     # puck in world coordinates
#     puck_xy = kart_xy + kart_to_puck
#     goal_to_puck = self.normalize(puck_xy - score_goal_xy)
#     # adjust for scoring
#     aim_vec = self.normalize(kart_to_puck + (goal_to_puck / 2))
#     theta = np.arccos(np.dot(kart_to_front, aim_vec))
#     signed_theta = -np.sign(np.cross(kart_to_front, aim_vec)) * theta
#     steer = 5*signed_theta

class HockeyPlayer:
    def __init__(self, player_id = 0):
        self.player_id = player_id
        self.kart = 'wilber'
        self.team = player_id % 2
        self.puck_detector = load_detector()
        self.puck_detector.eval()
        # self.vec_detector = load_vec()
        # self.vec_detector.eval()
        self.resize = torchvision.transforms.Resize([150, 200])

        self.first_frame = True
        self.im = None
        self.ax = plt.subplot(111)
        self.puck_draw = None
        self.new_puck_draw = None

        # num frames before we consider puck to be gone
        self.gone = 3
        self.gone_counter = 0

        # num frames before we consider puck to be back
        self.is_back_limit = 3
        self.is_back_counter = 0

        # number of frames before we need to rescue
        self.rescue_limit = 35
        self.rescue_counter = 0

        # number of frames to go forward while rescuing
        self.forward_counter = 0
        self.forward_steer = -1
        self.forward_limit = 25

        # Mode
        self.NORMAL_MODE = 0
        self.BACKUP_MODE = 1
        self.RESCUE_MODE = 2
        self.MODE = self.NORMAL_MODE

        # last 3 frames of which side puck was on (used for backing up in right direction)
        self.PUCK_SIDE_LEN = 3
        self.PUCK_SIDE = []

    def get_puck_coords(self, image):
        device = torch.device('cpu')
        I = Image.fromarray(image)
        I = self.resize(I)
        I = F.to_tensor(I)
        I = I[None, :]
        I = I.to(device)
        puck_data = self.puck_detector(I)
        puck_data = puck_data.detach().numpy()[0]
        return puck_data

    def puck_off_screen(self, screen_puck_xy):
        screen_puck_x = screen_puck_xy[0]
        screen_puck_y = screen_puck_xy[1]
        if (screen_puck_y > 225 and screen_puck_x > 150 and screen_puck_x < 250):
            return True
        else:
            return False

    def normalize(self, vec):
        return vec / np.linalg.norm(vec)

    def clamp(self, value):
        if value > 0:
            return min(1, value)
        else:
            return max(-1, value)

    def visualize(self, image, screen_puck_xy, new_puck_xy):
        if self.first_frame:
            self.im = self.ax.imshow(image)
            self.first_frame = False
        else:
            self.im.set_data(image)

        if self.puck_draw is not None:
            self.puck_draw.remove()
            self.new_puck_draw.remove()

        self.puck_draw = plt.Circle(screen_puck_xy, 10, ec='g', fill=False, lw=1.5)
        self.new_puck_draw = plt.Circle(new_puck_xy, 10, ec='r', fill=False, lw=1.5)
        self.ax.add_artist(self.puck_draw)
        self.ax.add_artist(self.new_puck_draw)

        plt.pause(0.001)
      
    def act(self, image, player_info):
        score_goal_xy = None
        if self.team == 0:
            score_goal_xy = GOAL_0
        else:
            score_goal_xy = GOAL_1

        front_xy = np.array(player_info.kart.front)[[0,2]]
        kart_xy = np.array(player_info.kart.location)[[0,2]]

        # location of puck on screen
        screen_puck_xy = self.get_puck_coords(image)

        kart_to_puck = screen_puck_xy - PLAYER_LOC
        kart_to_puck[1] = -kart_to_puck[1]
        dist_to_puck = np.linalg.norm(kart_to_puck)

        # player orientation vector
        kart_to_front = self.normalize(front_xy - kart_xy)
        kart_to_goal = self.normalize(score_goal_xy - kart_xy)
        theta = np.degrees(np.arccos(np.dot(kart_to_front, kart_to_goal)))
        turn_dir = np.sign(np.cross(kart_to_front, kart_to_goal))

        new_puck_xy = screen_puck_xy.copy()
        offset = min(30, theta)
        if dist_to_puck < 60:
            if turn_dir > 0:
                new_puck_xy[0] = new_puck_xy[0] + offset
            else:
                new_puck_xy[0] = new_puck_xy[0] - offset

        # set the player actions
        steer = self.clamp((new_puck_xy[0] - 200) / 20)
        accel = 0.5
        brake = False
        drift = False

        if self.MODE == self.NORMAL_MODE:
            # running tally of how many frames puck is off screen
            if self.puck_off_screen(screen_puck_xy):
                self.gone_counter += 1
            else:
                # side is -1 if on left, and +1 if on right
                side = 2 * ((screen_puck_xy[0] < 200) - 0.5)
                if len(self.PUCK_SIDE) == self.PUCK_SIDE_LEN:
                    self.PUCK_SIDE.pop(0)
                self.PUCK_SIDE.append(side)

                if self.gone_counter > 0:
                    self.gone_counter -= 1

            # puck has been gone for too long, turn on backup mode
            if self.gone_counter == self.gone:
                self.MODE = self.BACKUP_MODE
                self.is_back_counter = 0

        elif self.MODE == self.BACKUP_MODE:
            accel = 0
            brake = True

            if len(self.PUCK_SIDE) == 0:
                steer = -0.75
            elif np.mean(self.PUCK_SIDE) < 0:
                steer = -0.75
            else:
                steer = 0.75

            if self.puck_off_screen(screen_puck_xy):
                self.rescue_counter += 1
                if self.is_back_counter > 0:
                    self.is_back_counter -= 1
            else:
                self.is_back_counter += 1
                if self.rescue_counter > 0:
                    self.rescue_counter -= 1

            if self.is_back_counter == self.is_back_limit:
                self.gone_counter = 0
                self.rescue_counter = 0
                self.MODE = self.NORMAL_MODE
            elif self.rescue_counter == self.rescue_limit:
                self.gone_counter = 0
                self.rescue_counter = 0
                self.MODE = self.RESCUE_MODE
                self.forward_steer = 2 * ((random.uniform(-1, 1) > 0) - 0.5)

        elif self.MODE == self.RESCUE_MODE:
            accel = 1
            steer = self.forward_steer
            self.forward_counter += 1

            if self.forward_counter == self.forward_limit:
                self.forward_counter = 0
                self.MODE = self.NORMAL_MODE

        # visualize the controller in real time
        # if player_info.kart.id == 0:
        #     self.visualize(image, screen_puck_xy, new_puck_xy)

        action = {
            'steer': steer,
            'acceleration': accel,
            'brake': brake,
            'drift': drift,
            'nitro': False, 
            'rescue': False}

        return action