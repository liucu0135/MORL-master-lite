from __future__ import absolute_import, division, print_function
import numpy as np

class FruitTree(object):

    def __init__(self, depth=6):
        # the map of the deep sea treasure (convex version)
        self.reward_dim = 6
        self.tree_depth = depth # zero based depth
        branches = np.zeros((int(2 ** self.tree_depth - 1), self.reward_dim))
        # fruits = np.random.randn(2**self.tree_depth, self.reward_dim)
        # fruits = np.abs(fruits) / np.linalg.norm(fruits, 2, 1, True)
        # print(fruits*10)
        fruits = np.array(FRUITS[str(depth)])
        self.tree = np.concatenate(
            [
                branches,
                fruits
            ])

        # DON'T normalize
        self.max_reward = 10.0

        # state space specification: 2-dimensional discrete box
        self.state_spec = [['discrete', 1, [0, self.tree_depth]],
                           ['discrete', 1, [0, 2 ** self.tree_depth - 1]]]

        # action space specification: 0 left, 1 right
        self.action_spec = ['discrete', 1, [0, 2]]

        # reward specification: 2-dimensional reward
        # 1st: treasure value || 2nd: time penalty
        self.reward_spec = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]

        self.current_state = np.array([0, 0])
        self.terminal = False

    def get_ind(self, pos):
        return int(2 ** pos[0] - 1) + pos[1]

    def get_tree_value(self, pos):
        return self.tree[self.get_ind(pos)]

    def reset(self):
        '''
            reset the location of the submarine
        '''
        self.current_state = np.array([0, 0])
        self.terminal = False

    def step(self, action):
        '''
            step one move and feed back reward
        '''

        direction = {
            0: np.array([1, self.current_state[1]]),  # left
            1: np.array([1, self.current_state[1] + 1]),  # right
        }[action]

        self.current_state = self.current_state + direction

        reward = self.get_tree_value(self.current_state)
        if self.current_state[0] == self.tree_depth:
            self.terminal = True

        return self.current_state, reward, self.terminal
