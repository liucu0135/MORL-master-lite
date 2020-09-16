from __future__ import absolute_import, division, print_function
import argparse
import visdom
import torch
import numpy as np
from sklearn.manifold import TSNE

import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from envs.mo_env import MultiObjectiveEnv

parser = argparse.ArgumentParser(description='MORL-PLOT')
# CONFIG
parser.add_argument('--env-name', default='crp', metavar='ENVNAME',
                    help='environment to train on (default: tf): ft | ft5 | ft7')
parser.add_argument('--method', default='crl-envelope', metavar='METHODS',
                    help='methods: crl-naive | crl-envelope | crl-energy')
parser.add_argument('--model', default='conv', metavar='MODELS',
                    help='linear | cnn | cnn + lstm')
parser.add_argument('--gamma', type=float, default=0.99, metavar='GAMMA',
                    help='gamma for infinite horizonal MDPs')
# PLOT
parser.add_argument('--pltmap', default=False, action='store_true',
                    help='plot deep sea treasure map')
parser.add_argument('--pltpareto', default=False, action='store_true',
                    help='plot pareto frontier')
parser.add_argument('--pltcontrol', default=False, action='store_true',
                    help='plot control curve')
parser.add_argument('--pltdemo', default=False, action='store_true',
                    help='plot demo')
# LOG & SAVING
parser.add_argument('--save', default='crl/envelope/saved/', metavar='SAVE',
                    help='address for saving trained models')
parser.add_argument('--name', default='0', metavar='name',
                    help='specify a name for saving the model')
# Useless but I am too laze to delete them
parser.add_argument('--mem-size', type=int, default=10000, metavar='M',
                    help='max size of the replay memory')
parser.add_argument('--batch-size', type=int, default=256, metavar='B',
                    help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate')
parser.add_argument('--epsilon', type=float, default=0.3, metavar='EPS',
                    help='epsilon greedy exploration')
parser.add_argument('--epsilon-decay', default=False, action='store_true',
                    help='linear epsilon decay to zero')
parser.add_argument('--weight-num', type=int, default=32, metavar='WN',
                    help='number of sampled weights per iteration')
parser.add_argument('--episode-num', type=int, default=100, metavar='EN',
                    help='number of episodes for training')
parser.add_argument('--optimizer', default='Adam', metavar='OPT',
                    help='optimizer: Adam | RMSprop')
parser.add_argument('--update-freq', type=int, default=32, metavar='OPT',
                    help='optimizer: Adam | RMSprop')
parser.add_argument('--beta', type=float, default=0.01, metavar='BETA',
                    help='beta for evelope algorithm, default = 0.01')
parser.add_argument('--homotopy', default=False, action='store_true',
                    help='use homotopy optimization method')

args = parser.parse_args()
vis = visdom.Visdom()

assert vis.check_connection()

use_cuda = torch.cuda.is_available()
torch.cuda.device(0)
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class Paint_Booth():
    def __init__(self, args):
        vis = visdom.Visdom()
        assert vis.check_connection()
        FloatTensor = torch.cuda.FloatTensor
        LongTensor = torch.cuda.LongTensor
        ByteTensor = torch.cuda.ByteTensor
        Tensor = FloatTensor
        args = parser.parse_args()
        # setup the environment
        self.env = MultiObjectiveEnv(args.env_name)
        # get state / action / reward sizes
        state_size = len(env.state_spec)
        action_size = env.action_spec[2][1] - env.action_spec[2][0]
        reward_size = len(env.reward_spec)


        from crl.envelope.meta import MetaAgent
        from crl.envelope.models import get_new_model

        model = get_new_model(args.model, state_size, action_size, reward_size)
        agent = MetaAgent(model, args, is_train=True)

        state_size = len(env.state_spec)
        action_size = env.action_spec[2][1] - env.action_spec[2][0]
        reward_size = len(env.reward_spec)

        model = get_new_model(args.model, state_size, action_size, reward_size)
        dicts = torch.load("{}{}.pth.tar".format(args.save,
                                                 "m.{}_e.{}_n.{}".format(args.model, args.env_name, args.name)))
        model.load_state_dict(dicts['state_dict'])
        self.agent = MetaAgent(model, args, is_train=False)

    def run_one_episode(self, w):
        w = np.abs(w) / np.linalg.norm(w, ord=1)
        w_e = w / np.linalg.norm(w, ord=2)
        ttrw = np.array([0.0, 0.0])
        terminal = False
        self.env.reset()
        cnt = 0
        while not terminal:
            state = self.env.observe()
            mask = self.env.env.get_action_out_mask()
            action = self.agent.act(state, preference=torch.from_numpy(w).type(FloatTensor), mask=mask)
            next_state, reward, terminal = self.env.step(action)
            reward[0] = 1 - reward[0]
            reward[1] = self.env.env.get_distortion(absolute=True, tollerance=15) / 5
            if cnt > 300:
                terminal = True
            ttrw = ttrw + reward  # * np.power(args.gamma, cnt)
            cnt += 1
        # ttrw_w = w.dot(ttrw) * w_e
        return ttrw_w


################# Control Frontier #################
if __name__ == '__main__':
    args = parser.parse_args()
    # setup the environment
    env = MultiObjectiveEnv(args.env_name)
    torch.cuda.set_device(1)
    # get state / action / reward sizes
    state_size = len(env.state_spec)
    action_size = env.action_spec[2][1] - env.action_spec[2][0]
    reward_size = len(env.reward_spec)

    # generate an agent for initial training
    agent = None

    from crl.envelope.meta import MetaAgent
    from crl.envelope.models import get_new_model

    model = get_new_model(args.model, state_size, action_size, reward_size)
    agent = MetaAgent(model, args, is_train=True)


    state_size = len(env.state_spec)
    action_size = env.action_spec[2][1] - env.action_spec[2][0]
    reward_size = len(env.reward_spec)

    model = get_new_model(args.model, state_size, action_size, reward_size)
    dicts=torch.load("{}{}.pth.tar".format(args.save,
                                         "m.{}_e.{}_n.{}".format(args.model, args.env_name, args.name)))
    model.load_state_dict(dicts['state_dict'])
    agent = MetaAgent(model, args, is_train=False)

    # compute opt
    q_x = []
    q_y = []
    act_x = []
    act_y = []
    ws=np.arange(0,12)//2/5
    for i in range(1):  # $used to be 2000
        print('doing test {}'.format(i))
        w = [0,1]
        w = np.abs(w) / np.linalg.norm(w, ord=1)
        w_e = w / np.linalg.norm(w, ord=2)
        ttrw = np.array([0.0, 0.0])
        terminal = False
        env.reset()
        cnt = 0
        while not terminal:
            state = env.observe()
            mask = env.env.get_action_out_mask()
            action = agent.act(state, preference=torch.from_numpy(w).type(FloatTensor), mask=mask)
            next_state, reward, terminal = env.step(action)
            reward[0]=1-reward[0]
            reward[1]=env.env.get_distortion(absolute=True, tollerance=15)/5
            if cnt > 300:
                terminal = True
            ttrw = ttrw + reward #* np.power(args.gamma, cnt)
            cnt += 1
        ttrw_w = w.dot(ttrw) * w_e
        act_x.append(ttrw[0])
        act_y.append(ttrw[1])

    act_opt = dict(x=act_x,
                   y=act_y,
                   mode="markers",
                   type='custom',
                   marker=dict(
                       symbol="circle",
                       size=3),
                   name='policy')


    layout_opt = dict(title="FT Control Frontier - {} {}()".format(
        args.method, args.name),
        xaxis=dict(title='1st objective'),
        yaxis=dict(title='2nd objective'))
    vis._send({'data': [act_opt], 'layout': layout_opt})

