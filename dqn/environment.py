import cv2
from gym.spaces.box import Box
import gym
import random
import numpy as np
from .utils import rgb2gray, imresize


# def _process_frame42(frame):
#     frame = frame[34:34+160, :160]
#     # Resize by half, then down to 42x42 (essentially mipmapping). If
#     # we resize directly we lose pixels that, when mapped to 42x42,
#     # aren't close enough to the pixel boundary.
#     frame = cv2.resize(frame, (80, 80))
#     frame = cv2.resize(frame, (42, 42))
#     frame = frame.mean(2)
#     frame = frame.astype(np.float32)
#     frame *= (1.0 / 255.0)
#     frame = np.reshape(frame, [42, 42, 1])
#     return frame

# class AtariRescale42x42(vectorized.ObservationWrapper):
#     def __init__(self, env=None):
#         super(AtariRescale42x42, self).__init__(env)
#         self.observation_space = Box(0.0, 1.0, [42, 42, 1])

#     def _observation(self, observation_n):
#         return [_process_frame42(observation) for observation in observation_n]


class Environment(object):
  def __init__(self, config):
    self.env = gym.make(config.env_name)
    #rescale frames
    # self.env = AtariRescale42x42(self.env)

    screen_width, screen_height, self.action_repeat, self.random_start = \
        config.screen_width, config.screen_height, config.action_repeat, config.random_start

    self.display = config.display
    self.dims = (screen_width, screen_height)

    self._screen = None
    self.reward = 0
    self.terminal = True

  def new_game(self, from_random_game=False):
    if self.lives == 0:
      self._screen = self.env.reset()
    self._step(0)
    self.render()
    return self.screen, 0, 0, self.terminal

  def new_random_game(self):
    self.new_game(True)
    for _ in xrange(random.randint(0, self.random_start - 1)):
      self._step(0)
    self.render()
    return self.screen, 0, 0, self.terminal

  def _step(self, action):
    self._screen, self.reward, self.terminal, _ = self.env.step(action)

  def _random_step(self):
    action = self.env.action_space.sample()
    self._step(action)

  @ property
  def screen(self):
    return imresize(rgb2gray(self._screen)/255., self.dims)
    #return cv2.resize(cv2.cvtColor(self._screen, cv2.COLOR_BGR2YCR_CB)/255., self.dims)[:,:,0]

  @property
  def action_size(self):
    return self.env.action_space.n

  @property
  def lives(self):
    return self.env.ale.lives()

  @property
  def state(self):
    return self.screen, self.reward, self.terminal

  def render(self):
    if self.display:
      self.env.render()

  def after_act(self, action):
    self.render()

class GymEnvironment(Environment):
  def __init__(self, config):
    super(GymEnvironment, self).__init__(config)

  def act(self, action, is_training=True):
    cumulated_reward = 0
    start_lives = self.lives

    for _ in xrange(self.action_repeat):
      self._step(action)
      cumulated_reward = cumulated_reward + self.reward

      if is_training and start_lives > self.lives:
        cumulated_reward -= 1
        self.terminal = True

      if self.terminal:
        break

    self.reward = cumulated_reward

    self.after_act(action)
    return self.state

class SimpleGymEnvironment(Environment):
  def __init__(self, config):
    super(SimpleGymEnvironment, self).__init__(config)

  def act(self, action, is_training=True):
    self._step(action)

    self.after_act(action)
    return self.state
