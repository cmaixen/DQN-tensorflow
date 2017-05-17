"""Based on Code from https://github.com/tambetm/simple_dqn/blob/master/src/replay_memory.py"""
import os
import random
import logging
import sys
import math
import numpy as np

from .binary_heap import BinaryHeap

from .utils import save_npy, load_npy


class ReplayMemory:
  def __init__(self, config, model_dir):
    self.model_dir = model_dir

    self.cnn_format = config.cnn_format
    self.memory_size = config.memory_size
    self.actions = np.empty(self.memory_size, dtype = np.uint8)
    self.rewards = np.empty(self.memory_size, dtype = np.integer)
    self.screens = np.empty((self.memory_size, config.screen_height, config.screen_width), dtype = np.float16)
    self.terminals = np.empty(self.memory_size, dtype = np.bool)
    self.history_length = config.history_length
    self.dims = (config.screen_height, config.screen_width)
    self.batch_size = config.batch_size
    self.count = 0
    self.current = 0

    # pre-allocate prestates and poststates for minibatch
    self.prestates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.float16)
    self.poststates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.float16)

  def add(self, screen, reward, action, terminal):
    assert screen.shape == self.dims
    # NB! screen is post-state, after action and reward
    self.actions[self.current] = action
    self.rewards[self.current] = reward
    self.screens[self.current, ...] = screen
    self.terminals[self.current] = terminal
    self.count = max(self.count, self.current + 1)
    self.current = (self.current + 1) % self.memory_size

  def getState(self, index):
    assert self.count > 0, "replay memory is empy, use at least --random_steps 1"
    # normalize index to expected range, allows negative indexes
    index = index % self.count
    # if is not in the beginning of matrix
    if index >= self.history_length - 1:
      # use faster slicing
      return self.screens[(index - (self.history_length - 1)):(index + 1), ...]
    else:
      # otherwise normalize indexes and use slower list based access
      indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
      return self.screens[indexes, ...]

  def sample(self):
    # memory must include poststate, prestate and history
    assert self.count > self.history_length
    # sample random indexes
    indexes = []
    while len(indexes) < self.batch_size:
      # find random index 
      while True:
        # sample one index (ignore states wraping over 
        index = random.randint(self.history_length, self.count - 1)
        # if wraps over current pointer, then get new one
        if index >= self.current and index - self.history_length < self.current:
          continue
        # if wraps over episode end, then get new one
        # NB! poststate (last screen) can be terminal state!
        if self.terminals[(index - self.history_length):index].any():
          continue
        # otherwise use this index
        break
      
      # NB! having index first is fastest in C-order matrices
      self.prestates[len(indexes), ...] = self.getState(index - 1)
      self.poststates[len(indexes), ...] = self.getState(index)
      indexes.append(index)

    actions = self.actions[indexes]
    rewards = self.rewards[indexes]
    terminals = self.terminals[indexes]

    if self.cnn_format == 'NHWC':
      return np.transpose(self.prestates, (0, 2, 3, 1)), actions, \
        rewards, np.transpose(self.poststates, (0, 2, 3, 1)), terminals
    else:
      return self.prestates, actions, rewards, self.poststates, terminals

  def save(self):
    for idx, (name, array) in enumerate(
        zip(['actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates'],
            [self.actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates])):
      save_npy(array, os.path.join(self.model_dir, name))

  def load(self):
    for idx, (name, array) in enumerate(
        zip(['actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates'],
            [self.actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates])):
      array = load_npy(os.path.join(self.model_dir, name))



class PrioritizedReplayMemory(object):

    def __init__(self, config, model_dir):

      self.model_dir = model_dir
      self.cnn_format = config.cnn_format

      self.size = config.memory_size
      
      self.batch_size = config.batch_size
      self.replace_flag = True

      self.alpha = 0.7
      self.beta_zero = 0.5
      self.learn_start = config.learn_start

      # http://www.evernote.com/l/ACnDUVK3ShVEO7fDm38joUGNhDik3fFaB5o/
      self.total_steps = config.max_step

      self.index = 0
      self.record_size = 0
      self.isFull = False

      self._experience = {}
      self.priority_queue = BinaryHeap(self.size)

      
      self.beta_grad = (1 - self.beta_zero) / (self.total_steps - self.learn_start)

      #to check if there is enough in mem
      self.count = 0

    

    def build_distribution(self,size):
        # P(i) = (rank i) ^ (-alpha) / sum ((rank i) ^ (-alpha))
        pdf = list(
            map(lambda x: math.pow(x, -self.alpha), range(1, size + 1))
        )
        pdf_sum = math.fsum(pdf)
        power_law_distribution = list(map(lambda x: x / pdf_sum, pdf))

        return power_law_distribution

    def save(self, filename="memory_state"):
        data = np.array([
            self.size,
            self.replace_flag,
            self.alpha,
            self.beta_zero,
            self.batch_size,
            self.learn_start,
            self.total_steps,
            self.index,
            self.record_size,
            self.isFull,
            self._experience,
            self.priority_queue.priority_queue
        ])
        np.save(filename, data)

    def load(self, filename="memory_state"):
        data = np.load(filename)
        self.size,
        self.replace_flag,
        self.alpha,
        self.beta_zero,
        self.batch_size,
        self.learn_start,
        self.total_steps,
        self.index,
        self.record_size,
        self.isFull,
        self._experience,
        self.priority_queue.priority_queue = data
        self.priority_queue.reload()



    def fix_index(self):
        """
        get next insert index of our memory
        :return: index, int
        """
        if self.record_size <= self.size:
            self.record_size += 1
        if self.index % self.size == 0:
            self.isFull = True if len(self._experience) == self.size else False
            if self.replace_flag:
                self.index = 1
                return self.index
            else:
                sys.stderr.write('Experience replay buff is full and replace is set to FALSE!\n')
                return -1
        else:
            self.index += 1
            return self.index


    def add(self, s_t, reward, action, s_t_plus_1, terminal):

        experience = [s_t, reward, action, s_t_plus_1, terminal]

        insert_index = self.fix_index()
        if insert_index > 0:
            if insert_index in self._experience:
                del self._experience[insert_index]
            self._experience[insert_index] = experience
            # add to priority queue
            priority = self.priority_queue.get_max_priority()
            self.priority_queue.update(priority, insert_index)
            self.count +=1
            return True
        else:
            sys.stderr.write('Insert failed\n')
            return False

    def retrieve(self, indices):
        """
        get experience from indices

        """
        return [self._experience[v] for v in indices]

    def rebalance(self):
        """
        rebalance priority queue

        """
        self.priority_queue.balance_tree()

    def update_priority(self, indices, delta):
        """
        update priority according indices and deltas

        """
        for i in range(0, len(indices)):
            self.priority_queue.update(math.fabs(delta[i]), indices[i])
        self.rebalance()

    def sample(self, global_step):
        # we get the probalitly of selection a certain rank
        ranks = range(1, self.priority_queue.size + 1)
        distribution = self.build_distribution(self.priority_queue.size)
        # we define our ranks
        #selected k ranks based on our probability
        print(len(distribution))
        print(len(ranks))
        rank_list = np.random.choice(ranks,self.batch_size, p=distribution)
 
        # beta, increase by global_step, max 1
        beta = min(self.beta_zero + (global_step - self.learn_start - 1) * self.beta_grad, 1)
        # find all alpha pow, notice that pdf is a list, start from 0
        prob_i = [distribution[v - 1] for v in rank_list]
        # w = (N * P(i)) ^ (-beta) / max w
        w = np.power(np.array(prob_i) * self.size, -beta)
        w_max = max(w)
        w = np.divide(w, w_max)
        # rank list is priority id
        # convert to experience id
        rank_e_id = 0
        try:
            rank_e_id = self.priority_queue.priority_to_experience(rank_list)
        except:
            print('ERRROR')
        # get experience id according rank_e_id
        experiences = self.retrieve(rank_e_id)
        print(rank_e_id)
        #experience layout is [s_t, reward, action, s_t_plus_1, terminal]
        s_ts = np.array([experience[0] for experience in experiences])
        rewards = np.array([experience[1] for experience in experiences])
        actions = np.array([experience[2] for experience in experiences])
        s_t_plus_1s = np.array([experience[3] for experience in experiences])
        terminals = np.array([experience[4] for experience in experiences])


        if self.cnn_format == 'NHWC':

            #no transpose is needed because the we transpose it already we giving it as input.
            #the history object transposes it already
            return s_ts, actions, rewards ,s_t_plus_1s,terminals, w, rank_e_id
        else:
            return s_ts, actions, rewards ,s_t_plus_1s,terminals, w, rank_e_id

