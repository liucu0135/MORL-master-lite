from __future__ import absolute_import, division, print_function
import random
import torch
import copy
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from collections import namedtuple
from collections import deque

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class MetaAgent(object):
    '''
    (1) act: how to sample an action to examine the learning
        outcomes or explore the environment;
    (2) memorize: how to store observed observations in order to
        help learing or establishing the empirical model of the
        enviroment;
    (3) learn: how the agent learns from the observations via
        explicitor implicit inference, how to optimize the policy
        model.
    '''

    def __init__(self, model, args, is_train=False):
        self.model_ = model
        self.model = copy.deepcopy(model)
        self.is_train = is_train
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_delta = (args.epsilon - 0.05) / args.episode_num

        self.mem_size = args.mem_size
        self.batch_size = args.batch_size
        self.weight_num = args.weight_num

        self.beta            = args.beta
        self.beta_init       = args.beta
        self.homotopy        = args.homotopy
        self.beta_uplim      = 1.00
        self.tau             = 1000.
        self.beta_expbase    = float(np.power(self.tau*(self.beta_uplim-self.beta), 1./args.episode_num))
        self.beta_delta      = self.beta_expbase / self.tau

        self.preference_cov = np.identity(self.model.reward_size)
        self.preference_mean = np.ones(self.model.reward_size)


        self.trans_mem = deque()
        if args.env_name=='crp':
            self.trans = namedtuple('trans', ['s', 'a', 's_', 'r', 'd','m','m_'])
        else:
            self.trans = namedtuple('trans', ['s', 'a', 's_', 'r', 'd'])

        self.priority_mem = deque()

        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model_.parameters(), lr=args.lr)
        elif args.optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model_.parameters(), lr=args.lr)

        self.w_kept = None
        self.update_count = 0
        self.update_freq = args.update_freq

        if self.is_train:
            self.model.train()
        if use_cuda:
            self.model.cuda()
            self.model_.cuda()

    def act(self, state, preference=None, mask=None):
        # random pick a preference if it is not specified
        if preference is None:
            if self.w_kept is None:
                self.w_kept = torch.randn(self.model_.reward_size)
                # self.w_kept /=torch.from_numpy(np.sqrt(self.preference_mean)+0.0001)
                # self.w_kept = torch.from_numpy(
                # # 1 / (0.0001+np.abs(np.random.multivariate_normal(np.zeros(self.model.reward_size), self.preference_cov))))
                # 1 / (0.0001+np.abs(np.random.multivariate_normal(self.preference_mean, self.preference_cov))))
                # self.w_kept += torch.Tensor([0,10])

                self.w_kept = (torch.abs(self.w_kept) / \
                               torch.norm(self.w_kept, p=1)).type(FloatTensor)
                print('exploration preference: {}'.format(self.w_kept))

            preference = self.w_kept
        state = torch.from_numpy(state).type(FloatTensor)

        _, Q = self.model_(
            Variable(state.unsqueeze(0)),
            Variable(preference.unsqueeze(0)))

        Q = Q.view(-1, self.model_.reward_size)

        Q = torch.mv(Q.data, preference)
        Q0=Q
        if mask is not None:
            Q=Q-(1-mask.cuda())*torch.max(torch.abs(Q)*2+100)
        action = Q.max(0)[1].cpu().numpy()
        action = int(action)
        if not mask[action]:
            print('mask is not working, mask:{}, action: {}, Q:{}, Q0:{}'.format(mask, action, Q, Q0))
        if self.is_train and (len(self.trans_mem) < self.batch_size or \
                              torch.rand(1)[0] < self.epsilon):
            action = np.random.choice(self.model_.action_size, 1)[0]
            while not mask[action]:
                action = np.random.choice(self.model_.action_size, 1)[0]
            action = int(action)
        if not mask[action]:
            print('mask is not working')
        return action

    def memorize(self, state, action, next_state, reward, terminal, mask=None, next_mask=None):
        if mask is None:
            self.trans_mem.append(self.trans(
            torch.from_numpy(state).type(FloatTensor),  # state
            action,  # action
            torch.from_numpy(next_state).type(FloatTensor),  # next state
            torch.from_numpy(reward).type(FloatTensor),  # reward
            terminal))  # terminal
        else:
            self.trans_mem.append(self.trans(
                torch.from_numpy(state).type(FloatTensor),  # state
                action,  # action
                torch.from_numpy(next_state).type(FloatTensor),  # next state
                torch.from_numpy(reward).type(FloatTensor),  # reward
                terminal,  # terminal
                mask,
                next_mask, ))   # masks

        # randomly produce a preference for calculating priority
        # preference = self.w_kept
        preference = torch.randn(self.model_.reward_size)
        preference = (torch.abs(preference) / torch.norm(preference, p=1)).type(FloatTensor)
        state = torch.from_numpy(state).type(FloatTensor)

        _, q = self.model_(Variable(state.unsqueeze(0), requires_grad=False),
                           Variable(preference.unsqueeze(0), requires_grad=False))

        q = q[0, action].data  # this <action> is already masked
        wq = preference.dot(q)

        wr = preference.dot(torch.from_numpy(reward).type(FloatTensor))
        if not terminal:
            next_state = torch.from_numpy(next_state).type(FloatTensor)
            hq, _ = self.model_(Variable(next_state.unsqueeze(0), requires_grad=False),
                                Variable(preference.unsqueeze(0), requires_grad=False), next_mask=next_mask)
            hq = hq.data[0]
            whq = preference.dot(hq)
            p = abs(wr + self.gamma * whq - wq)
        else:
            print('pref_mean:{}, pref_cov:{}'.format(self.preference_mean, self.preference_cov))
            self.w_kept = None
            # if self.epsilon_decay:
            #     # self.epsilon -= self.epsilon_delta
            #     if self.epsilon>0.001:
            #         self.epsilon *= 0.95# self.beta_delta
            # if self.homotopy:
            #     if self.beta<0.95:
            #         self.beta += 0.01# self.beta_delta
                    # self.beta_delta = (self.beta-self.beta_init)*self.beta_expbase+self.beta_init-self.beta
            p = abs(wr - wq)
        p += 1e-5
        # p = 1e-5  #blocked
        self.priority_mem.append(
            p
        )
        if len(self.trans_mem) > self.mem_size:
            self.trans_mem.popleft()
            self.priority_mem.popleft()

    def sample(self, pop, pri, k):
        pri = np.array(pri).astype(np.float)
        inds = np.random.choice(
            range(len(pop)), k,
            replace=False,
            p=pri / pri.sum()
        )
        return [pop[i] for i in inds]

    def actmsk(self, num_dim, index):
        mask = ByteTensor(num_dim).zero_()
        mask[index] = 1
        return mask.unsqueeze(0)

    def nontmlinds(self, terminal_batch):
        mask = ByteTensor(terminal_batch)
        inds = torch.arange(0, len(terminal_batch)).type(LongTensor)
        inds = inds[mask.eq(0)]
        return inds

    def learn(self):
        if len(self.trans_mem) > self.batch_size:

            self.update_count += 1

            action_size = self.model_.action_size
            reward_size = self.model_.reward_size

            minibatch = self.sample(self.trans_mem, self.priority_mem, self.batch_size)
            # minibatch = random.sample(self.trans_mem, self.batch_size)
            batchify = lambda x: list(x) * self.weight_num
            state_batch = batchify(map(lambda x: x.s.unsqueeze(0), minibatch))
            action_batch = batchify(map(lambda x: LongTensor([x.a]), minibatch))
            reward_batch = batchify(map(lambda x: x.r.unsqueeze(0), minibatch))
            next_state_batch = batchify(map(lambda x: x.s_.unsqueeze(0), minibatch))
            terminal_batch = batchify(map(lambda x: x.d, minibatch))
            if len(minibatch[0])==7:
                next_mask_batch = batchify(map(lambda x: x.m_.unsqueeze(0), minibatch))


            # update cov
            reward_data=torch.cat(reward_batch, dim=0).detach().cpu().numpy()
            # self.preference_cov=np.cov(reward_data, rowvar=False)

            # update mean
            self.preference_mean=np.mean(np.abs(reward_data), axis=0)

            w_batch = np.random.randn(self.weight_num, reward_size)
            # w_batch=w_batch/np.repeat(np.expand_dims(np.sqrt(self.preference_mean)+0.0001,axis=0),self.weight_num,axis=0)
            # w_batch = (np.abs(np.random.multivariate_normal(self.preference_mean, self.preference_cov, size=self.weight_num)))
            # w_batch=w_batch[:,::-1]
            # w_batch = 1 / (0.0001+np.abs(np.random.multivariate_normal(np.zeros(self.model.reward_size), self.preference_cov, size=self.weight_num)))
            # w_batch[:,1]+= 10
            w_batch = np.abs(w_batch) / \
                      np.linalg.norm(w_batch, ord=1, axis=1, keepdims=True)
            w_batch = torch.from_numpy(w_batch.repeat(self.batch_size, axis=0)).type(FloatTensor)

            __, Q = self.model_(Variable(torch.cat(state_batch, dim=0)),
                                Variable(w_batch), w_num=self.weight_num)
            # HQ, _    = self.model_(Variable(torch.cat(next_state_batch, dim=0), volatile=True),q'q'q'q'q'q'q'q'q'q'q
            # 					  Variable(w_batch, volatile=True), w_num=self.weight_num)
            _, DQ = self.model(Variable(torch.cat(next_state_batch, dim=0), requires_grad=False),
                               Variable(w_batch, requires_grad=False))
            w_ext = w_batch.unsqueeze(2).repeat(1, action_size, 1)
            w_ext = w_ext.view(-1, self.model.reward_size)
            _, tmpQ = self.model_(Variable(torch.cat(next_state_batch, dim=0), requires_grad=False),
                                  Variable(w_batch, requires_grad=False))


            tmpQ = tmpQ.view(-1, reward_size)
            # print(torch.bmm(w_ext.unsqueeze(1),
            # 			    tmpQ.data.unsqueeze(2)).view(-1, action_size))
            act = torch.bmm(Variable(w_ext.unsqueeze(1), requires_grad=False),
                            tmpQ.unsqueeze(2)).view(-1, action_size)
            act=act-(1-torch.cat(next_mask_batch, dim=0).cuda())*torch.max(torch.abs(act)+1)
            act=act.max(1)[1]

            HQ = DQ.gather(1, act.view(-1, 1, 1).expand(DQ.size(0), 1, DQ.size(2))).squeeze()

            nontmlmask = self.nontmlinds(terminal_batch)
            with torch.no_grad():
                Tau_Q = Variable(torch.zeros(self.batch_size * self.weight_num,
                                             reward_size).type(FloatTensor))
                Tau_Q[nontmlmask] = self.gamma * HQ[nontmlmask]
                # Tau_Q.volatile = False
                Tau_Q += Variable(torch.cat(reward_batch, dim=0))

            actions = Variable(torch.cat(action_batch, dim=0))

            Q = Q.gather(1, actions.view(-1, 1, 1).expand(Q.size(0), 1, Q.size(2))
                         ).view(-1, reward_size)
            Tau_Q = Tau_Q.view(-1, reward_size)

            wQ = torch.bmm(Variable(w_batch.unsqueeze(1)),
                           Q.unsqueeze(2)).squeeze()

            wTQ = torch.bmm(Variable(w_batch.unsqueeze(1)),
                            Tau_Q.unsqueeze(2)).squeeze()

            # loss = F.mse_loss(Q.view(-1), Tau_Q.view(-1))
            loss = self.beta * F.mse_loss(wQ.view(-1), wTQ.view(-1))
            loss += (1-min(self.beta_uplim,self.beta)) * F.mse_loss(Q.view(-1), Tau_Q.view(-1))

            self.optimizer.zero_grad()
            loss.backward()
            for param in self.model_.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

            if self.update_count % self.update_freq == 0:
                self.model.load_state_dict(self.model_.state_dict())

            return loss.data

        return 0.0

    def reset(self):
        self.w_kept = None
        if self.epsilon_decay:
            if self.epsilon>0.001:
                self.epsilon *=0.8
                # print('eps:{}'.format(self.epsilon))
            # self.epsilon -= self.epsilon_delta
        if self.homotopy:
            if self.beta<0.95:
                self.beta += 0.02# self.beta_delta
            # self.beta += self.beta_delta
            # self.beta_delta = (self.beta-self.beta_init)*self.beta_expbase+self.beta_init-self.beta

    def predict(self, probe, initial_state=None):
        if initial_state is None:
            return self.model(Variable(FloatTensor([0, 0]).unsqueeze(0), requires_grad=False),
                          Variable(probe.unsqueeze(0), requires_grad=False))
        else:
            return self.model(Variable(FloatTensor(initial_state).unsqueeze(0), requires_grad=False),
                              Variable(probe.unsqueeze(0), requires_grad=False))

    def save(self, save_path, model_name):
        torch.save({'state_dict': self.model.state_dict()}, "{}{}.pth.tar".format(save_path, model_name))


    def find_preference(
            self,
            w_batch,
            target_batch,
            pref_param):

        with torch.no_grad():
            w_batch = FloatTensor(w_batch)
            target_batch = FloatTensor(target_batch)

        # compute loss
        pref_param = FloatTensor(pref_param)
        pref_param.requires_grad = True
        sigmas = FloatTensor([0.001]*len(pref_param))
        dist = torch.distributions.normal.Normal(pref_param, sigmas)
        pref_loss = dist.log_prob(w_batch).sum(dim=1) * target_batch

        self.optimizer.zero_grad()
        # Total loss
        loss = pref_loss.mean()
        loss.backward()
        
        eta = 1e-3
        pref_param = pref_param + eta * pref_param.grad
        pref_param = simplex_proj(pref_param.detach().cpu().numpy())
        # print("update prefreence parameters to", pref_param)

        return pref_param


# projection to simplex
def simplex_proj(x):
    y = -np.sort(-x)
    sum = 0
    ind = []
    for j in range(len(x)):
        sum = sum + y[j]
        if y[j] + (1 - sum) / (j + 1) > 0:
            ind.append(j)
        else:
            ind.append(0)
    rho = np.argmax(ind)
    delta = (1 - (y[:rho+1]).sum())/(rho+1)
    return np.clip(x + delta, 0, 1)
