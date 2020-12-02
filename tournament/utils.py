import pystk
import numpy as np
import pathlib
import PIL.Image
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import torch

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from agent.player import HockeyPlayer

HACK_DICT = dict()

class Player:
    def __init__(self, player, team=0):
        self.player = player
        self.team = team

    @property
    def config(self):
        return pystk.PlayerConfig(controller=pystk.PlayerConfig.Controller.PLAYER_CONTROL, kart=self.player.kart, team=self.team)
    
    def __call__(self, image, player_info):
        return self.player.act(image, player_info)

class DummyPlayer:
    def __init__(self, team=0):
        self.team = team

    @property
    def config(self):
        return pystk.PlayerConfig(
            controller=pystk.PlayerConfig.Controller.AI_CONTROL,
            team=self.team)
    
    def __call__(self, image, player_info):
        return dict()

FIRST = True
IM = None
BACKUP = False
FRAMES = 0

class OraclePlayer:
    kart = ""
    
    def __init__(self, player_id = 0):
        #all_players = ['adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley', 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux', 'wilber', 'xue']
        self.player_id = player_id

        # players with the largest max_steer_angle
        if player_id == 0:
            self.kart = 'wilber'
        if player_id == 1:
            self.kart = 'hexley'
        if player_id == 2:
            self.kart = 'konqi'
        if player_id == 3:
            self.kart = 'xue'
        
        self.team = player_id % 2
        
    def act(self, image, player_info):
        """
        :param image: numpy array of shape (300, 400, 3)
        :param player_info: pystk.Player object for the current kart.
        return: Dict describing the action
        """
        global FIRST, IM, BACKUP

        front = np.array(HACK_DICT['kart'].front)[[0, 2]]
        kart = np.array(HACK_DICT['kart'].location)[[0, 2]]
        puck = np.array(HACK_DICT['state'].soccer.ball.location)[[0, 2]]

        u = front - kart
        u = u / np.linalg.norm(u)

        v = puck - kart
        v = v / np.linalg.norm(v)

        theta = np.arccos(np.dot(u, v))
        signed_theta = -np.sign(np.cross(u, v)) * theta

        steer = 20 * signed_theta
        accel = 0.5
        brake = False
        drift = False

        STUCK = False
        if (player_info.kart.velocity[0] < 0.01):
          FRAMES += 1
        else:
          FRAMES = 0
        
        threshold = 50
        if (FRAMES >= threshold):
          STUCK = True
          FRAMES = 0

        if (STUCK):
          v = -4*u
          theta = np.arccos(np.dot(u, v))
          signed_theta = -np.sign(np.cross(u, v)) * theta
          steer = 20 * signed_theta

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
            print('angle: ', "{:.2f}".format(np.degrees(signed_theta)))
            print('loc: ' + str(kart))
            plt.pause(0.001)
        

        action = {
            'steer': steer,
            'acceleration': accel,
            'brake': brake,
            'drift': drift,
            'nitro': False, 
            'rescue': False}

        return action

class Tournament:
    _singleton = None

    def __init__(self, players, screen_width=400, screen_height=300, track='icy_soccer_field'):
        assert Tournament._singleton is None, "Cannot create more than one Tournament object"
        Tournament._singleton = self

        HACK_DICT.clear()

        self.graphics_config = pystk.GraphicsConfig.hd()
        self.graphics_config.screen_width = screen_width
        self.graphics_config.screen_height = screen_height
        pystk.init(self.graphics_config)

        self.race_config = pystk.RaceConfig(num_kart=len(players), track=track, mode=pystk.RaceConfig.RaceMode.SOCCER)
        self.race_config.players.pop()
        
        self.active_players = []
        for p in players:
            if p is not None:
                self.race_config.players.append(p.config)
                self.active_players.append(p)
        
        self.k = pystk.Race(self.race_config)

        self.k.start()
        self.k.step()

    def play(self, save=None, max_frames=50, save_callback=None):
        state = pystk.WorldState()
        
        if save is not None:
            import PIL.Image
            if not os.path.exists(save):
                os.makedirs(save)

        # turn on interactive mode for controller visualization
        plt.ion()

        for t in range(max_frames):
            print('\rframe %d' % t, end='\r')

            state.update()
            list_actions = []

            for i, p in enumerate(self.active_players):
                HACK_DICT['race'] = self.k
                HACK_DICT['render_data'] = self.k.render_data[i]
                HACK_DICT['kart'] = state.karts[i]
                HACK_DICT['state'] = state
                
                #print('kart: ', state.karts[i].name, '\t\t max_steer: ', "{:.4f}".format(state.karts[i].max_steer_angle))

                player = state.players[i]
                image = np.array(self.k.render_data[i].image)
                
                action = pystk.Action()
                player_action = p(image, player)
                for a in player_action:
                    setattr(action, a, player_action[a])
                
                list_actions.append(action)

                if save is not None:
                    im = PIL.Image.fromarray(image)
                    #draw = ImageDraw.Draw(im)
                    #draw.text((0, 0),"Sample Text",(255,255,255))
                    im.save(os.path.join(save, 'player%02d_%05d.png' % (i, t)))

            for i, action in enumerate(list_actions):
                HACK_DICT['player_%d' % i] = {
                    'steer': action.steer,
                    'acceleration': action.acceleration
                    }

            if save_callback is not None:
                save_callback(self.k, state, t, HACK_DICT)

            s = self.k.step(list_actions)
            if not s:  # Game over
                break

        if save is not None:
            import subprocess
            for i, p in enumerate(self.active_players):
                dest = os.path.join(save, 'player%02d' % i)
                output = 'videos/_player%02d.mp4' % i
                subprocess.call(['ffmpeg', '-y', '-framerate', '10', '-i', dest + '_%05d.png', output])
                
        if hasattr(state, 'soccer'):
            return state.soccer.score
        return state.soccer_score

    def close(self):
        self.k.stop()
        del self.k

class DataCollector(object):
    def __init__(self, destination):
        self.images = list()
        self.destination = pathlib.Path(destination)
        self.destination.mkdir(exist_ok=True)

        self.image = os.path.join(destination, 'images')
        if not os.path.exists(self.image): 
            os.mkdir(self.image)

        self.puck = os.path.join(destination, 'puck')
        if not os.path.exists(self.puck): 
            os.mkdir(self.puck)

        self.action = os.path.join(destination, 'actions')
        if not os.path.exists(self.action): 
            os.mkdir(self.action)

    def save_frame(self, race, state, t, hack_dict):
        #if t % 2 == 0:
        #    self._debug(race, state)

        # player
        for i in range(4):
            mask = (race.render_data[i].instance == 134217729)

            has_puck = False
            for row in mask:
                for b in row:
                    if b:
                        has_puck = True
                        break
            
            if not has_puck:
                continue

            output_path = '%s/%d_%06d.png' % (self.puck, i, t)
            Image.fromarray(mask).save(output_path)

            image = race.render_data[i].image       # np uint8 (h, w, 3) [0, 255]
            output_path = '%s/%d_%06d.png' % (self.image, i, t)
            Image.fromarray(image).save(output_path)

            action = hack_dict['player_%d' % i]
            output_path = '%s/%d_%06d.txt' % (self.action, i, t)
            pathlib.Path(output_path).write_text('%.3f %.3f' % (action['steer'], action['acceleration']))

    def visualize_player(self, race, state, i):
        # 255 * mask.astype(np.uint8) -> (h, w) [0-255]
        # mask[..., None] -> (h, w, 1)
        # mask.repeat(3, -1) -> (h, w, 3)
        mask = (race.render_data[i].instance == 134217729)
        mask = (255 * mask.astype(np.uint8))[..., None].repeat(3, -1)

        return np.hstack((race.render_data[i].image, mask))

    # def _debug(self, race, state):
    #     visualizations = [DataCollector.visualize_player(race, state, i) for i in range(4)]

    #     row1 = np.hstack((visualizations[0], visualizations[1]))
    #     row2 = np.hstack((visualizations[2], visualizations[3]))
    #     grid = np.vstack((row1, row2))

    #     self.images.append(grid[::2, ::2])

    # def colab_show(self):
    #     if len(self.images) == 0:
    #         return

    #     from moviepy.editor import ImageSequenceClip
    #     from IPython.display import display

    #     display(ImageSequenceClip(self.images, fps=10).ipython_display(width=720, autoplay=True, loop=True))


def run(agents, dest):
    players = []

    for i, player in enumerate(agents):
        if player == 'AI':
            players.append(DummyPlayer(i % 2))
        else:
            players.append(Player(player(i), i % 2))

    data_collector = DataCollector(dest)
        
    tournament = Tournament(players)
    score = tournament.play(max_frames=250, save_callback=data_collector.save_frame)

    print('Final score', score)

def test(agents, dest=None):
    players = []

    for i, player in enumerate(agents):
        if player == 'AI':
            players.append(DummyPlayer(i % 2))
        else:
            players.append(Player(player(i), i % 2))
        
    tournament = Tournament(players)
    #score = tournament.play(save=dest, max_frames=200)
    score = tournament.play(max_frames=1000)

    print('Final score', score)

if __name__ == '__main__':
    # Collect an episode.
     run([OraclePlayer, OraclePlayer, OraclePlayer, OraclePlayer], 'data')
    # test([OraclePlayer, OraclePlayer, OraclePlayer, OraclePlayer], 'test')
    # test([HockeyPlayer, OraclePlayer])