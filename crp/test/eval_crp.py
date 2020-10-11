from __future__ import absolute_import, division, print_function
import argparse
import visdom
import torch
import numpy as np
from sklearn.manifold import TSNE
from read_result import read_result
import pandas as pd

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
parser.add_argument('--gamma', type=float, default=1, metavar='GAMMA',
                    help='gamma for infinite horizonal MDPs')
parser.add_argument('--bench_csv', default='./test/bench.csv',
                    help='location for benchmark csv file')
parser.add_argument('--num_orders', type=int, default=1000, metavar='M',
                    help='max size of the replay memory')
parser.add_argument('--cc', default=False, action='store_true')
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
parser.add_argument('--save', default='crl/envelope/saved2/', metavar='SAVE',
                    help='address for saving trained models')

parser.add_argument('--name', default='norm_ex_norm_learn005_sample_shaped_nc', metavar='name',
                    help='specify a name for saving the model')
# Useless but I am too laze to delete them
parser.add_argument('--mem_size', type=int, default=1000, metavar='M',
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
# torch.cuda.device(1)
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor




# apply gamma
# env_ind = {'ft':'6', 'ft5':'5', 'ft7':'7'}[args.env_name]


def matrix2lists(MATRIX):
    X, Y = [], []
    for x in list(MATRIX[:, 0]):
        X.append(float(x))
    for y in list(MATRIX[:, 1]):
        Y.append(float(y))
    return X, Y


def find_in(A, B, eps=0.2):
    # find element of A in B with a tolerance of relative err of eps.
    cnt1, cnt2 = 0.0, 0.0
    for a in A:
        for b in B:
            if eps > 0.001:
              if np.linalg.norm(a - b, ord=1) < eps*np.linalg.norm(b):
                  cnt1 += 1.0
                  break
            else:
              if np.linalg.norm(a - b, ord=1) < 0.5:
                  cnt1 += 1.0
                  break
    for b in B:
        for a in A:
            if eps > 0.001:
              if np.linalg.norm(a - b, ord=1) < eps*np.linalg.norm(b):
                  cnt2 += 1.0
                  break
            else:
              if np.linalg.norm(a - b, ord=1) < 0.5:
                  cnt2 += 1.0
                  break
    return cnt1, cnt2


################# Control Frontier #################
if __name__ == '__main__':
    args = parser.parse_args()
    # setup the environment
    env = MultiObjectiveEnv(args)
    # torch.cuda.set_device(1)
    # get state / action / reward sizes
    state_size = len(env.state_spec)
    action_size = env.action_spec[2][1] - env.action_spec[2][0]
    reward_size = len(env.reward_spec)
    # record the result to csv
    record_data=[]
    # generate an agent for initial training
    agent = None
    if args.method == 'crl-naive':
        from crl.naive.meta import MetaAgent
        from crl.naive.models import get_new_model
    elif args.method == 'crl-envelope':
        from crl.envelope.meta import MetaAgent
        from crl.envelope.models import get_new_model
    elif args.method == 'crl-energy':
        from crl.energy.meta import MetaAgent
        from crl.energy.models import get_new_model

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
    opt_x = []
    opt_y = []
    q_x = []
    q_y = []
    act_x = []
    act_y = []
    ws=np.arange(0,110)/100
    opt_x,opt_y = read_result(args.bench_csv)
    for i in range(100):  # $used to be 2000
        print('doing test {}'.format(i))

        w=i/100.0
        w=[w, 1-w]


        # w = np.random.randn(2)
        w = np.abs(w) / np.linalg.norm(w, ord=1)
        # w = np.random.dirichlet(np.ones(2))
        w_e = w / np.linalg.norm(w, ord=2)
        # if args.method == 'crl-naive' or args.method == 'crl-envelope':
        #     hq, _ = agent.predict(torch.from_numpy(w).type(FloatTensor))
        # elif args.method == 'crl-energy':
        #     hq, _ = agent.predict(torch.from_numpy(w).type(FloatTensor), alpha=1e-5)
        # realc = real_sol.dot(w).max() * w_e
        # qc = w_e
        # if args.method == 'crl-naive':
        #     qc = hq.data[0] * w_e
        # elif args.method == 'crl-envelope':
        #     qc = w.dot(hq.data.cpu().numpy().squeeze()) * w_e
        # elif args.method == 'crl-energy':
        #     qc = w.dot(hq.data.cpu().numpy().squeeze()) * w_e
        ttrw = np.array([0.0, 0.0])
        terminal = False
        env.reset()
        cnt = 0
        while not terminal:
            state = env.observe()
            mask = env.env.get_action_out_mask()
            action = agent.act(state, preference=torch.from_numpy(w).type(FloatTensor), mask=mask)
            next_state, reward, terminal = env.step(action)
            reward[0]=env.env.stepcc
            # reward[1]=env.env.get_distortion()
            # reward[1]=-reward[1]
            reward[1]=env.env.get_distortion(absolute=True, tollerance=0)/10
            if cnt > args.num_orders:
                terminal = True
            ttrw = ttrw + reward #* np.power(args.gamma, cnt)
            cnt += 1
        # ttrw_w = w.dot(ttrw) * w_e

        # q_x.append(qc[0])
        # q_y.append(qc[1])
        act_x.append(ttrw[0])
        act_y.append(ttrw[1])
        record_data.append(ttrw)
    trace_opt = dict(x=act_x,
                     y=act_y,
                     mode="markers",
                     type='custom',
                     marker=dict(
                         symbol="circle",
                         size=3),
                     name='real')

    act_opt = dict(x=opt_x,
                   y=opt_y,
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
    vis._send({'data': [trace_opt, act_opt], 'layout': layout_opt})
    df=pd.DataFrame.from_records(record_data)
    df.to_csv('result_{}.csv'.format(args.name))