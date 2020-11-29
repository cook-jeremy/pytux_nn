import numpy as np


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
    #added
    def __init__(self, player_id = 0):
        """
        Set up a soccer player.
        The player_id starts at 0 and increases by one for each player added. You can use the player id to figure out your team (player_id % 2), or assign different roles to different agents.
        """
        self.player_id = player_id
        self.kart = 'tux'
        self.team = player_id % 2
    #added   
    def act(self, image, player_info):
        """
        Set the action given the current image
        :param image: numpy array of shape (300, 400, 3)
        :param player_info: pystk.Player object for the current kart.
        return: Dict describing the action
        """
        action = {'acceleration': 0, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}
       

        front = np.float32(HACK_DICT['kart'].front)[[0, 2]]
        kart = np.float32(HACK_DICT['kart'].location)[[0, 2]]
        puck = np.float32(HACK_DICT['state'].soccer.ball.location)[[0, 2]]

        u = front - kart
        u = u / np.linalg.norm(u)

        v = puck - kart
        v = v / np.linalg.norm(v)

        theta = np.arccos(np.dot(u, v))
        signed_theta = -np.sign(np.cross(u, v)) * theta

        return {
            'steer': 20 * signed_theta,
            'acceleration': 0.5,
            'brake': False,
            'drift': np.degrees(theta) > 60,
            'nitro': False, 'rescue': False}

