from __future__ import absolute_import, division, print_function
import argparse
import visdom
import torch
import numpy as np
from sklearn.manifold import TSNE
from read_result import read_result

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
parser.add_argument('--save', default='crl/envelope/saved2/', metavar='SAVE',
                    help='address for saving trained models')
parser.add_argument('--name', default='uni_ex_uni_learn_sample_shaped_ccm', metavar='name',
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
    env = MultiObjectiveEnv(args.env_name)
    # torch.cuda.set_device(1)
    # get state / action / reward sizes
    state_size = len(env.state_spec)
    action_size = env.action_spec[2][1] - env.action_spec[2][0]
    reward_size = len(env.reward_spec)

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
    ws=np.arange(1,10)/10
    real_sol = read_result()
    opt_x=real_sol[:,0].tolist()
    opt_y=real_sol[:,1].tolist()
    for i in range(100):  # $used to be 2000
        print('doing test {}'.format(i))
        # w = [0,1]
        # w = [1-ws[i],ws[i]]

        if i<20:
            w=[0.8,0.2]
        elif i<40:
            w=[0.6,0.4]
        elif i<60:
            w = [0.5, 0.5]
        elif i<80:
            w = [0.4, 0.6]
        else:
            w=[0.2, 0.8]
        # w = np.random.randn(2)
        w = np.abs(w) / np.linalg.norm(w, ord=1)
        # w = np.random.dirichlet(np.ones(2))
        w_e = w / np.linalg.norm(w, ord=2)
        # if args.method == 'crl-naive' or args.method == 'crl-envelope':
        #     hq, _ = agent.predict(torch.from_numpy(w).type(FloatTensor))
        # elif args.method == 'crl-energy':
        #     hq, _ = agent.predict(torch.from_numpy(w).type(FloatTensor), alpha=1e-5)
        realc = real_sol.dot(w).max() * w_e
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
            reward[0]=-reward[0]
            # reward[1]=env.env.get_distortion()
            # reward[1]=-reward[1]
            reward[1]=env.env.get_distortion(absolute=True, tollerance=0)/10
            if cnt > 1000:
                terminal = True
            ttrw = ttrw + reward #* np.power(args.gamma, cnt)
            cnt += 1
        ttrw_w = w.dot(ttrw) * w_e

        # q_x.append(qc[0])
        # q_y.append(qc[1])
        act_x.append(ttrw[0])
        act_y.append(ttrw[1])
    trace_opt = dict(x=act_x[:20]+act_x[40:60]+act_x[80:],
                     y=act_y[:20]+act_y[40:60]+act_y[80:],
                     mode="markers",
                     type='custom',
                     marker=dict(
                         symbol="circle",
                         size=3),
                     name='real')

    act_opt = dict(x=act_x[20:40]+act_x[60:80],
                   y=act_y[20:40]+act_y[60:80],
                   mode="markers",
                   type='custom',
                   marker=dict(
                       symbol="circle",
                       size=3),
                   name='policy')

    # q_opt = dict(x=q_x,
    #              y=q_y,
    #              mode="markers",
    #              type='custom',
    #              marker=dict(
    #                  symbol="circle",
    #                  size=1),
    #              name='predicted')

    # ## quantitative evaluation
    # policy_loss = 0.0
    # predict_loss = 0.0
    # TEST_N = 5000.0
    # for i in range(int(TEST_N)):
    #     w = np.random.randn(6)
    #     w = np.abs(w) / np.linalg.norm(w, ord=1)
    #     # w = np.random.dirichlet(np.ones(2))
    #     w_e = w / np.linalg.norm(w, ord=2)
    #     if args.method == 'crl-naive' or args.method == 'crl-envelope':
    #         hq, _ = agent.predict(torch.from_numpy(w).type(FloatTensor))
    #     elif args.method == 'crl-energy':
    #         hq, _ = agent.predict(torch.from_numpy(w).type(FloatTensor), alpha=1e-5)
    #     realc = real_sol.dot(w).max() * w_e
    #     qc = w_e
    #     if args.method == 'crl-naive':
    #         qc = hq.data[0] * w_e
    #     elif args.method == 'crl-envelope':
    #         qc = w.dot(hq.data.cpu().numpy().squeeze()) * w_e
    #     elif args.method == 'crl-energy':
    #         qc = w.dot(hq.data.cpu().numpy().squeeze()) * w_e
    #     ttrw = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    #     terminal = False
    #     env.reset()
    #     cnt = 0
    #     while not terminal:
    #         state = env.observe()
    #         action = agent.act(state, preference=torch.from_numpy(w).type(FloatTensor))
    #         next_state, reward, terminal = env.step(action)
    #         if cnt > 30:
    #             terminal = True
    #         ttrw = ttrw + reward * np.power(args.gamma, cnt)
    #         cnt += 1
    #     ttrw_w = w.dot(ttrw) * w_e
    #
    #     base = np.linalg.norm(realc, ord=2)
    #     policy_loss += np.linalg.norm(realc - ttrw_w, ord=2)/base
    #     predict_loss += np.linalg.norm(realc - qc, ord=2)/base
    #
    # policy_loss /= TEST_N / 100
    # predict_loss /= TEST_N / 100


    # print("discrepancies (100*err): policy-{}|predict-{}".format(policy_loss, predict_loss))

    # layout_opt = dict(title="FT Control Frontier - {} {}({:.3f}|{:.3f})".format(
    #     args.method, args.name, policy_loss, predict_loss),
    #     xaxis=dict(title='1st objective'),
    #     yaxis=dict(title='2nd objective'))
    layout_opt = dict(title="FT Control Frontier - {} {}()".format(
        args.method, args.name),
        xaxis=dict(title='1st objective'),
        yaxis=dict(title='2nd objective'))
    vis._send({'data': [trace_opt, act_opt], 'layout': layout_opt})

################# Pareto Frontier #################

if args.pltpareto:

    # setup the environment
    env = MultiObjectiveEnv(args.env_name)

    # generate an agent for plotting
    agent = None
    if args.method == 'crl-naive':
        from crl.naive.meta import MetaAgent
    elif args.method == 'crl-envelope':
        from crl.envelope.meta import MetaAgent
    elif args.method == 'crl-energy':
        from crl.energy.meta import MetaAgent
    model = torch.load("{}{}.pkl".format(args.save,
                                         "m.{}_e.{}_n.{}".format(args.model, args.env_name, args.name)))
    agent = MetaAgent(model, args, is_train=False)

    # compute recovered Pareto
    act = []

    # predicted solution
    pred = []

    for i in range(200):  #2000
        w = np.random.randn(6)
        w = np.abs(w) / np.linalg.norm(w, ord=1)
        # w = np.random.dirichlet(np.ones(6))
        ttrw = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        terminal = False
        env.reset()
        cnt = 0
        if args.method == "crl-envelope":
            hq, _ = agent.predict(torch.from_numpy(w).type(FloatTensor))
            pred.append(hq.data.cpu().numpy().squeeze() * 1.0)
        elif args.method == "crl-energy":
            hq, _ = agent.predict(torch.from_numpy(w).type(FloatTensor), alpha=1e-5)
            pred.append(hq.data.cpu().numpy().squeeze() * 1.0)
        while not terminal:
            state = env.observe()
            action = agent.act(state, preference=torch.from_numpy(w).type(FloatTensor))
            next_state, reward, terminal = env.step(action)
            if cnt > 50:
                terminal = True
            ttrw = ttrw + reward * np.power(args.gamma, cnt)
            cnt += 1

        act.append(ttrw)

    act = np.array(act)
    cnt1, cnt2 = find_in(act, FRUITS, 0.0)
    act_precition = cnt1 / len(act)
    act_recall = cnt2 / len(FRUITS)
    act_f1 = 2 * act_precition * act_recall / (act_precition + act_recall)
    pred_f1 = 0.0
    pred_precition = 0.0
    pred_recall = 0.0

    if not pred:
        pred = act
    else:
        pred = np.array(pred)
        cnt1, cnt2 = find_in(pred, FRUITS)
        pred_precition = cnt1 / len(pred)
        pred_recall = cnt2 / len(FRUITS)
        if pred_precition > 1e-8 and pred_recall > 1e-8:
            pred_f1 = 2 * pred_precition * pred_recall / (pred_precition + pred_recall)

    FRUITS = np.tile(FRUITS, (30, 1))
    ALL = np.concatenate([FRUITS, act, pred])
    ALL = TSNE(n_components=2).fit_transform(ALL)
    p1 = FRUITS.shape[0]
    p2 = FRUITS.shape[0] + act.shape[0]

    fruit = ALL[:p1, :]
    act = ALL[p1:p2, :]
    pred = ALL[p2:, :]

    fruit_x, fruit_y = matrix2lists(fruit)
    act_x, act_y = matrix2lists(act)
    pred_x, pred_y = matrix2lists(pred)

    # Create and style traces
    trace_pareto = dict(x=fruit_x,
                        y=fruit_y,
                        mode="markers",
                        type='custom',
                        marker=dict(
                            symbol="circle",
                            size=10),
                        name='Pareto')

    act_pareto = dict(x=act_x,
                      y=act_y,
                      mode="markers",
                      type='custom',
                      marker=dict(
                          symbol="circle",
                          size=10),
                      name='Recovered')

    pred_pareto = dict(x=pred_x,
                       y=pred_y,
                       mode="markers",
                       type='custom',
                       marker=dict(
                           symbol="circle",
                           size=3),
                       name='Predicted')

    layout = dict(title="FT Pareto Frontier - {} {}({:.3f}|{:.3f})".format(
        args.method, args.name, act_f1, pred_f1))

    print("Precison: policy-{}|prediction-{}".format(act_precition, pred_precition))
    print("Recall: policy-{}|prediction-{}".format(act_recall, pred_recall))
    print("F1: policy-{}|prediction-{}".format(act_f1, pred_f1))

    # send to visdom
    if args.method == "crl-naive":
        vis._send({'data': [trace_pareto, act_pareto], 'layout': layout})
    elif args.method == "crl-envelope":
        vis._send({'data': [trace_pareto, act_pareto, pred_pareto], 'layout': layout})
    elif args.method == "crl-energy":
        vis._send({'data': [trace_pareto, act_pareto, pred_pareto], 'layout': layout})

################# HEATMAP #################

if args.pltmap:
    FRUITS_EMB = TSNE(n_components=2).fit_transform(FRUITS)
    X, Y = matrix2lists(FRUITS_EMB)
    trace_fruit_emb = dict(x=X, y=Y,
                           mode="markers",
                           type='custom',
                           marker=dict(
                               symbol="circle",
                               size=10),
                           name='Pareto')
    layout = dict(title="FRUITS")
    vis._send({'data': [trace_fruit_emb], 'layout': layout})
