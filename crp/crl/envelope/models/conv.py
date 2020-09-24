from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor



class EnvelopeConvCQN(torch.nn.Module):
    '''
        Linear Controllable Q-Network, Envelope Version
    '''

    def __init__(self, state_size, action_size, reward_size, para_reduce=1):
        super(EnvelopeConvCQN, self).__init__()
        self.include_last=True
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.color_num = 10
        self.m = 10
        self.c = 10
        self.fcnum = (7 * (8 + 2))
        self.num_lane = 7
        self.ccm=None


        # setting a couple of layers
        conv0 = [
            nn.Conv2d(self.m, 8, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.ReLU(),
        ]

        conv1 = [
            nn.Conv2d(4 + self.include_last * 2, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
        ]

        fc0 = [
            nn.Linear(8 * self.fcnum, 128),
            nn.ReLU(),
        ]
        fc1 = [
            nn.Linear(32 * self.c * self.m, 128),
            nn.ReLU(),
        ]
        fc_fuse = [
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256-2),
            nn.ReLU(),
        ]
        self.fca1 = nn.Linear(64+self.m, self.color_num)
        self.fca2 = nn.Linear(64+self.m, self.color_num)
        # self.fca=nn.Linear(128+self.c, c)
        self.fcv1 = nn.Linear(64 + 1, 1)
        self.fcv2 = nn.Linear(64 + 1, 1)
        self.conv0 = nn.Sequential(*conv0)
        self.conv1 = nn.Sequential(*conv1)
        self.fc1 = nn.Sequential(*fc1)
        self.fc0 = nn.Sequential(*fc0)
        self.fc_fuse = nn.Sequential(*fc_fuse)


    # def forward(self, state, preference, w_num=1):
    #     s_num = int(preference.size(0) / w_num)
    #     x = torch.cat((state, preference), dim=1)
    #     x = x.view(x.size(0), -1)
    #     x = F.relu(self.affine1(x))
    #     x = F.relu(self.affine2(x))
    #     x = F.relu(self.affine3(x))
    #     x = F.relu(self.affine4(x))
    #     q = self.affine5(x)
    #
    #     q = q.view(q.size(0), self.action_size, self.reward_size)
    #
    #     hq = self.H(q.detach().view(-1, self.reward_size), preference, s_num, w_num)
    #
    #     return hq, q

    def forward(self, s, preference, w_num=1, next_mask=None):
        #   360=5*5*8+16*10
        s_num = int(preference.size(0) / w_num)
        a_num = self.color_num
        state = s[:,:10*7*10].view(-1,10,7,self.m)
        tab_mc = s[:,10*7*10:].view(-1,self.c+6,self.c)
        # if not training means the batch size doesn't exist, unsqueezing the data to match dimension
        if not self.training:
            state = state.unsqueeze(0)
            tab = tab_mc[:self.m, :self.c].unsqueeze(0).unsqueeze(0)
            if self.include_last:
                dist_alert = tab_mc[-6, :self.c].repeat(self.m, 1).unsqueeze(0).unsqueeze(0)
                last = tab_mc[-5, :self.c].repeat(self.m, 1).unsqueeze(0).unsqueeze(0)
            else:
                last = tab_mc[-5, :self.c].repeat(self.m, 1).mm(self.ccm).unsqueeze(0).unsqueeze(0)

            step = tab_mc[-4, 0].unsqueeze(0).unsqueeze(0)
            hist = tab_mc[-3, :self.m].repeat(1, self.c, 1).unsqueeze(0).transpose(3, 2)
            hist2 = tab_mc[-2, :self.m].repeat(1, self.c, 1).unsqueeze(0).transpose(3, 2)
            hist3 = tab_mc[-1, :self.m].repeat(1, self.c, 1).unsqueeze(0).transpose(3, 2)
        else:
            tab = tab_mc[:, :self.m, :self.c].unsqueeze(1)
            if self.include_last:
                dist_alert = tab_mc[:, -6, :self.c].unsqueeze(1).repeat(1, self.m, 1).unsqueeze(1)
                last = tab_mc[:, -5, :self.c].unsqueeze(1).repeat(1, self.m, 1).unsqueeze(1)
            else:
                last = tab_mc[:, -5, :self.c].mm(self.ccm).unsqueeze(1).repeat(1, self.m, 1).unsqueeze(1)
            step = tab_mc[:, -4, 0].unsqueeze(1)
            hist = tab_mc[:, -3, :self.m].unsqueeze(1).repeat(1, self.c, 1).unsqueeze(1).transpose(3, 2)
            hist2 = tab_mc[:, -2, :self.m].unsqueeze(1).repeat(1, self.c, 1).unsqueeze(1).transpose(3, 2)
            hist3 = tab_mc[:, -1, :self.m].unsqueeze(1).repeat(1, self.c, 1).unsqueeze(1).transpose(3, 2)


        hist = torch.min(tab, hist)
        hist2 = torch.min(tab, hist2)
        hist3 = torch.min(tab, hist3)
        x = torch.cat((tab, hist, hist2, hist3, last, dist_alert), dim=1)
        # x3= torch.cat((torch.sum(hist, dim=2).squeeze(1),dist_alert[:,0,0,:]), dim=1)

        x1 = self.conv1(x)
        x1 = self.fc1(x1.view(-1, 32 * self.c * self.m))
        x2 = self.conv0(state)
        x2 = self.fc0(x2.view(-1, 8 * self.fcnum))
        x = self.fc_fuse(torch.cat((x1, x2), dim=1))
        x = torch.cat((x, preference), dim=1)
        v1 = self.fcv1(torch.cat((x[:, :64], step), dim=1))
        a1 = self.fca1(torch.cat((x[:, 64:128],torch.sum(hist, dim=2).squeeze(1)),dim=1))
        # a=self.fca(torch.cat((x[:,128:], dist_alert[:,0,0,:]), dim=1))
        m1 = torch.mean(a1, dim=1).unsqueeze(1).repeat(1, a1.shape[1])
        a1 = a1 - m1
        q1 = a1 + v1
        v2 = self.fcv2(torch.cat((x[:, 128:64+128], step), dim=1))
        a2 = self.fca2(torch.cat((x[:, 64+128:],dist_alert[:,0,0,:]),dim=1))
        # a=self.fca(torch.cat((x[:,128:], dist_alert[:,0,0,:]), dim=1))
        m2 = torch.mean(a2, dim=1).unsqueeze(1).repeat(1, a2.shape[1])
        a2 = a2- m2
        q2 = a2 + v2
        q=torch.stack((q1,q2),dim=2)
        if next_mask is not None:
            next_mask=(1-next_mask).unsqueeze(1).unsqueeze(0).expand(-1,-1,w_num).cuda()
            hq = q-next_mask*torch.max(torch.abs(q)+1)
            hq = self.H(hq.detach().view(-1, self.reward_size), preference, s_num, w_num)
        else:
            hq=None

        return hq, q

    def H(self, Q, w, s_num, w_num, next_mask=None):



        # mask for reordering the batch
        mask = torch.cat(
            [torch.arange(i, s_num * w_num + i, s_num)
             for i in range(s_num)]).type(LongTensor)
        reQ = Q.view(-1, self.action_size * self.reward_size
                     )[mask].view(-1, self.reward_size)

        # extend Q batch and preference batch
        reQ_ext = reQ.repeat(w_num, 1)
        w_ext = w.unsqueeze(2).repeat(1, self.action_size * w_num, 1)
        w_ext = w_ext.view(-1, self.reward_size)

        # produce the inner products
        prod = torch.bmm(reQ_ext.unsqueeze(1), w_ext.unsqueeze(2)).squeeze()

        # mask for take max over actions and weights
        prod = prod.view(-1, self.action_size * w_num)
        inds = prod.max(1)[1]
        mask = ByteTensor(prod.size()).zero_()
        mask.scatter_(1, inds.data.unsqueeze(1), 1)
        mask = mask.view(-1, 1).repeat(1, self.reward_size).bool()

        # get the HQ
        HQ = reQ_ext.masked_select(Variable(mask)).view(-1, self.reward_size)

        return HQ

    # def H_(self, Q, w, s_num, w_num):
    #     reQ = Q.view(-1, self.reward_size)
    #
    #     # extend preference batch
    #     w_ext = w.unsqueeze(2).repeat(1, self.action_size, 1).view(-1, 2)
    #
    #     # produce hte inner products
    #     prod = torch.bmm(reQ.unsqueeze(1), w_ext.unsqueeze(2)).squeeze()
    #
    #     # mask for take max over actions
    #     prod = prod.view(-1, self.action_size)
    #     inds = prod.max(1)[1]
    #     mask = ByteTensor(prod.size()).zero_()
    #     mask.scatter_(1, inds.data.unsqueeze(1), 1)
    #     mask = mask.view(-1, 1).repeat(1, self.reward_size)
    #
    #     # get the HQ
    #     HQ = reQ.masked_select(Variable(mask)).view(-1, self.reward_size)
    #
    #     return HQ


