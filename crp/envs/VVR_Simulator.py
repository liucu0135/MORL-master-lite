import numpy as np
import torch
import pandas as pd

from .VVR_Bank import VVR_Bank as Bank


#  Target:    def step(), return state, reward, terminal
# To do:
#   1. create self.state:    cpu state storage
#   2. create self.terminal: count remaining orders inside sim object
#   3. create self.reward:   calculate reward inside sim object
class VVR_Simulator():
    def __init__(self, args=None, num_color=14, num_model=10, capacity=7*8*6//10, num_lanes=7,lane_length=8, cc_file='envs/cost.csv',color_dist_file='envs/total_orders.csv', cc=False):
        self.num_color = num_color
        self.num_model = num_model
        self.num_lanes = num_lanes
        self.target_color=-1
        self.stepcc=0
        self.lane_length = lane_length
        self.rewards = [1, 0, -1]  # [0]for unchange, [1]for change, [2]for error
        self.capacity = capacity
        self.terminal = False
        self.args=args
        # if args:
        #     self.orders_num=args.num_orders
        #     cc=args.cc
        # else:
        #     self.orders_num=300
        # self.orders_num=500




        self.state_len=self.num_model*self.num_lanes*(self.lane_length+2)+(max(self.num_color,self.num_model)+6)*(max(self.num_color,self.num_model))
        self.reward_spec = [[0, 100], [0, 1000]]
        self.state_spec = [['discrete', 1, [0, self.num_lanes-10]]]*self.state_len
        self.action_spec = ['discrete', 1, [0, self.num_color]]
        self.ccm=self.read_cc_matrix(cc_file)
        if not cc:
            self.ccm=np.ones_like(self.ccm)-np.identity(self.ccm.shape[0])
            color_dist_file=None
            print('nc is used')
            print(self.ccm)
        self.color_dist_file = color_dist_file
        if color_dist_file :
            print('cc and color dist are used')
            print(self.ccm)
            self.color_dist=self.read_color_dist(color_dist_file)
            self.model_dist = self.color_dist[:self.num_model] / sum(self.color_dist[:self.num_model])
        self.reset()

    def read_in_orders(self, path):
        orders=pd.read_csv(path,encoding='unicode_escape')
        models=orders['Model']
        colors=orders['Color']
        count=0
        # mdict={}
        # for k in colors:
        #     if k not in mdict:
        #         mdict[k]=count
        #         count+=1
        models=list(models)
        colors=[ord(c)-65 for c in colors]
        return models, colors




    def reset(self):
        # if self.args.eval:
        self.fix_models, self.fix_colors = self.read_in_orders(self.args.exact_orders)
        # else:
        #     if np.random.uniform(0,1)>0.5:
        #         self.fix_models, self.fix_colors=self.read_in_orders(self.args.exact_orders)
        #     else:
        #         self.fix_models, self.fix_colors = self.read_in_orders(self.args.exact_orders2)
        self.start_sequencec = self.fix_colors[::-1]
        self.start_sequencem = self.fix_models[::-1]
        self.orders_num=len(self.fix_models)
        # self.start_sequencec = np.random.choice(range(self.num_color), self.orders_num).tolist()
        # self.start_sequencem = np.random.choice(range(self.num_model), self.orders_num).tolist()
        # if self.color_dist_file is not None:
        #     self.start_sequencec = np.random.choice(range(self.num_color), self.orders_num, p=self.color_dist).tolist()
        # else:
        #     self.start_sequencec = np.random.choice(range(self.num_color), self.orders_num).tolist()
        # if self.color_dist_file is not None:
        #     self.start_sequencem = np.random.choice(range(self.num_model), self.orders_num, p=self.model_dist).tolist()
        # else:
        #     self.start_sequencem = np.random.choice(range(self.num_model), self.orders_num).tolist()

        self.bank = Bank(fix_init=True, num_of_colors=self.num_model, sequence=self.start_sequencem,
                         num_of_lanes=self.num_lanes,
                         lane_length=self.lane_length)
        self.plan_list = {k: [c, m] for k, c, m in
                          zip(range(len(self.start_sequencec)), self.start_sequencec, self.start_sequencem)}
        self.mc_tab = np.zeros((self.num_model, self.num_color), dtype=np.int)
        self.job_list = np.zeros(self.num_model)
        self.last_color = -1
        self.nc = 0
        self.dist = 0

        for i in range(self.capacity):
            self.BBA_rule_step_in()
        ######### new components ###########
        self.terminal=False
        self.current_state = self.get_out_tensor()





    def read_cc_matrix(self,cc_file):
        ccm=pd.read_csv(cc_file)
        ccm=ccm.to_numpy(dtype=np.float,copy=True)[:,1:]/7.6
        return ccm

    def read_color_dist(self, color_file):
        dist = pd.read_csv(color_file)
        dist = dist.to_numpy(dtype=np.float, copy=True).squeeze()
        dist=dist/sum(dist)
        return dist

    def get_out_tensor(self, gpu=False, cc=False):
        num_tensor=max(self.num_color,self.num_model)
        in_tensor = torch.zeros(self.num_model)
        mct=torch.zeros(num_tensor,num_tensor)
        last_tensor = torch.zeros(num_tensor)
        dist_alert_tensor = torch.zeros(num_tensor).float()

        m_hist= torch.zeros(num_tensor)
        m_hist2= torch.zeros(num_tensor)
        m_hist3= torch.zeros(num_tensor)
        m_hist[:self.num_model] = torch.Tensor(self.bank.front_hist()).float()
        m_hist2[:self.num_model] = torch.Tensor(self.bank.front_hist(scope=2)).float()
        m_hist3[:self.num_model] = torch.Tensor(self.bank.front_hist(scope=3)).float()
        steps = torch.ones(num_tensor).float()*len(self.bank.out_queue)/100
        if self.last_color>-1:
            last_tensor[self.last_color]=1


        for c in range(self.num_color):
            t=self.search_color_dist(c)
            dist_alert_tensor[c]=t-len(self.start_sequencec)-self.capacity
                # d=dist_alert_tensor
                # dist_alert_tensor=torch.zeros_like(d)
                # dist_alert_tensor[np.argmax(d)]=1



        #
        # for i in range(len(self.start_sequencec)+self.capacity, self.orders_num):
        #     if i in range(sorted(self.plan_list.keys(), reverse=True)):
        #         latency=max(0,i-len(self.start_sequencec)-self.capacity)
        #         if ctemp[self.plan_list[i][0]]==self.plan_list[i][1]:
        #             if dist_alert_tensor[self.plan_list[i][0]]<i-len(self.start_sequencec)+self.capacity:
        #                 dist_alert_tensor[self.plan_list[i][0]]=
        #             continue

        if len(self.bank.in_queue):
            in_m=self.bank.in_queue[0]
            in_tensor[in_m]=1
            in_tensor=in_tensor.unsqueeze(1).unsqueeze(1)
            in_tensor=in_tensor.repeat(1,self.num_lanes,1)
            in_tensor=torch.cat((torch.FloatTensor(self.bank.state),in_tensor), dim=2)
        mct[:self.num_model,:self.num_color]=torch.FloatTensor(self.mc_tab)
        # if self.num_color<self.num_model:
        #     mct = torch.cat((mct, torch.zeros(self.num_model,self.num_model-self.num_color)),dim=1)
        mct=torch.cat((mct,dist_alert_tensor.unsqueeze(0)),dim=0)
        mct=torch.cat((mct,last_tensor.unsqueeze(0)),dim=0)
        mct=torch.cat((mct,steps.unsqueeze(0)),dim=0)
        mct=torch.cat((mct,m_hist.unsqueeze(0)),dim=0)
        mct=torch.cat((mct,m_hist2.unsqueeze(0)),dim=0)
        mct=torch.cat((mct,m_hist3.unsqueeze(0)),dim=0)

        if gpu:
            return in_tensor.cuda().float(), mct.cuda()
        else:
            return in_tensor.float(), mct

    def VVR_rule_out(self):
        if self.last_color==-1:
            last=self.target_color
        else:
            last=self.last_color
        self.job_list, last_color = self.find_job_list(last, alert=True) # select available models

        # select a model
        lane_dists=np.zeros(self.num_lanes)
        for l in range(self.num_lanes):
            model = self.bank.front_view()[l]
            if (self.job_list[model]) and (model > -1):
                lane_dists[l]=self.find_nearest_order(model, last_color)
        l=np.argmax(lane_dists)
        if np.max(lane_dists)==0:
            print("empty")
        return self.release(l, last)

    def step(self, action, step=0):
        last=self.last_color
        self.set_target(action)
        self.step_forward_out_semi_rl()
        reward=[]
        self.stepcc = self.ccm[last, self.last_color]
        if not last==self.last_color:
            reward.append(1-self.ccm[last,self.last_color])
        else:
            reward.append(1)
        # reward.append(-self.get_distortion())
        # step=1

        # reward.append((2-self.get_distortion()/100))
        reward.append((1-self.get_distortion(tollerance=20)/100))
        # reward.append(-self.get_distortion(absolute=True, tollerance=0)/10)
        self.current_state=self.observe()
        if len(self.start_sequencec)<2:
            self.terminal=True
        return self.current_state, np.stack(reward), self.terminal

    def observe(self):
        s1, s2=self.get_out_tensor()
        s1=s1.view(-1,1)    # model x num_lanes x length+2
        s2=s2.view(-1,1)    # model+6  x  color
        s=torch.cat((s1,s2),dim=0) #
        # l=len(s)-self.num_model*self.num_color*(self.lane_length+2)-(max(self.num_color,self.num_model)+6)*(max(self.num_color,self.num_model))
        return s.squeeze().numpy()

    def set_target(self,c):
        self.last_color=c

    def find_job_list(self, last_color, alert=False):
        m_hist = self.bank.front_hist()
        jl = np.minimum(self.mc_tab[:, last_color], m_hist)
        if (sum(jl) > 0) or (not alert):
            return jl, last_color
        else:
            print("not posiible")
            print(self.bank.get_view_state())
            print(self.mc_tab)
            return jl, last_color

    def rule_select_color(self):
        m_hist = self.bank.front_hist()
        max_jl = np.zeros(self.num_model)
        color = None
        for c in range(self.num_color):
            jl = np.minimum(self.mc_tab[:, c], m_hist)
            if self.ccm is not None:
                jl=jl.astype(np.float64)/self.ccm[c,self.last_color]
            if sum(jl) > sum(max_jl):
                max_jl = jl.copy()
                color = c

        if color == None:
            print("not posiible")
            print(self.bank.get_view_state())
            print(self.mc_tab)
        return color


    def search_color(self):# try to find if there are any car in "last_color"
        m_hist = self.bank.front_hist()
        last_color=self.last_color
        jl = np.minimum(self.mc_tab[:, last_color], m_hist)
        return (last_color > -1) and (sum(jl) > 0)



    def get_action_out_mask(self):
        mask=torch.zeros(self.num_color)
        m_hist = self.bank.front_hist()
        for c in range(self.num_color):
            jl = np.minimum(self.mc_tab[:, c], m_hist)
            if sum(jl) > 0:
                mask[c]=1
        return mask

    def action_out(self,a):
        c = a % self.num_color
        m = (a - c) // self.num_color

        fv = self.bank.front_view()
        for l in range(self.num_lanes):
            if fv[l]==m:
                return self.release(l,c)


    # def step_forward_out_semi_rl(self):
    #     last = self.last_color
    #     if 1:
    #         if self.BBA_rule_step_in():
    #             if self.VVR_rule_out():
    #                 if last==-1:
    #                     return self.rewards[0]
    #                 else:
    #                     if last==self.last_color:
    #                         return self.rewards[0]
    #                     else:
    #                         print('color is changed')
    #                         return self.rewards[1]
    #             else:
    #                 return self.rewards[2]
    #         else:
    #             return self.rewards[2]

    def step_forward_out_semi_rl(self):
        last = self.last_color
        if len(self.start_sequencec):
            if self.BBA_rule_step_in():
                if self.VVR_rule_out():
                    if last==-1:
                        return self.rewards[0]
                    else:
                        if last==self.last_color:
                            return self.rewards[0]
                        else:
                            print('color is changed')
                            return self.rewards[1]
                else:
                    return self.rewards[2]
            else:
                return self.rewards[2]
        else:
            if self.VVR_rule_out():
                if last==-1:
                    return self.rewards[0]
                else:
                    if last==self.last_color:
                        return self.rewards[0]
                    else:
                        print('color is changed')
                        return self.rewards[1]
            else:
                return self.rewards[2]


    def BBA_rule_step_in(self):
        color = self.start_sequencec[-1]
        model = self.bank.in_queue[-1]
        c = self.bank.check_rear(model)
       # cb = self.bank.check_color_block(model)

        if max(c) > 0:
            r = self.bank.insert(np.argmax(c))
            self.mc_tab[model, color] += r
        else:
            c = self.bank.check_all(model)
            if max(c) > 0:
                r = self.bank.insert(np.argmax(c))
                self.mc_tab[model, color] += r
            else:
                r = self.bank.insert(np.argmax(-self.bank.cursors))
                self.mc_tab[model, color] += r
        if r:
            self.start_sequencec.pop()
        return r


    def find_nearest_order(self, m,c):
        for k in range(self.orders_num):
            if self.orders_num-k in self.plan_list:
                if self.plan_list[self.orders_num-k][0]==c and self.plan_list[self.orders_num-k][1]==m:
                    return self.orders_num-k
        # print('nearest order not found')
        return -1

    # def search_model(self, color):
    #     job_list, color = self.find_job_list(color)
    #     lane_dists = np.zeros(self.num_lanes)
    #     for l in range(self.num_lanes):
    #         model = self.bank.front_view()[l]
    #         if (job_list[model]) and (model > -1):
    #             lane_dists[l] = self.find_nearest_order(model, color)
    #     l = np.argmax(lane_dists)
    #     return l
    def search_color_dist(self, color):
        job_list, color = self.find_job_list(color)
        lane_dists = np.zeros(self.num_lanes)
        for l in range(self.num_lanes):
            model = self.bank.front_view()[l]
            if (job_list[model]) and (model > -1):
                lane_dists[l] = self.find_nearest_order(model, color)
        l = np.max(lane_dists)
        return l

    def release(self, lane, color):
        model = self.bank.front_view()
        model = model[lane]
        if model < 0:
            return False  # return false if the lane is empty
        if self.mc_tab[model, color] > 0:
            if not self.bank.release(lane):
                return False  # return false if the release fails
            self.mc_tab[model, color] -= 1
            if not (color == self.last_color):
                self.cc += self.ccm[color, self.last_color]
                self.last_cc=self.ccm[color, self.last_color]
                self.last_color = color
            k=self.find_nearest_order(model,color)
            self.plan_list.pop(k)
            self.released_order_index=k
            self.last_model=model
            return True  # return True if the release success
        else:
            return False  # return false if the corresponding model and color was not needed anymore


    def get_distortion(self, absolute=False, tollerance=0, color=0, model=0, sequence=None):
        if sequence is None:
            # absolute gives the actual KPI of tardness:  the deviation of each order in the sequence with a tolerance
            if absolute:
                n=np.maximum((self.released_order_index-len(self.start_sequencec)-self.capacity)-tollerance, 0)
                return n
            # self.plan_list is the orders that are delayed
            dist_count=0
            # $len(self.start_sequencec) + self.capacity$ is the number of order that should be painted now
            # i counts to $self.orders_num+1$ which is the total number of all generated orders in the simulator
            # this for loop iterates all orders which should be painted the
            for i in range(len(self.start_sequencec)+self.capacity, self.orders_num+1):
                if i in self.plan_list:  # if an order should be painted is not painted
                    dist_count+=i-len(self.start_sequencec)-self.capacity  # count the delay
            return dist_count
        else:
            n = np.maximum((self.released_order_index - len(self.start_sequencec) - self.capacity) - tollerance, 0)
            return n