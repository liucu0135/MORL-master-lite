from sequence_reader import get_sequence
import numpy as np
import os
import pandas as pd

class PBS():
    def __init__(self, r1, r2, tar, capacity=2):
        self.container=[]
        self.capacity= capacity
        self.dists=[]
        self.cursor=0
        self.in1=r1
        self.in2=r2
        # self.in1=tar[::2,:]
        # self.in2=tar[1::2,:]
        self.pending_list=[]
        self.released=np.zeros(len(tar))
        self.target=tar

        for _ in range(capacity):
            self.fill_in()

    def get_distance(self, order):
        # for i in range(len(self.target)):
        for i in range(len(self.target[:self.cursor])):
            if np.array_equal(self.target[i], order) and self.label[i]==0:
                self.label[i]=1
                # return max(0, self.cursor-i+1)
                return self.cursor-i+1
        # print('order not found!')
        return 0


    def fill_in(self):
        released=0
        if len(self.in1)>0:
            self.container.append(self.in1[0])
            self.in1=np.delete(self.in1, 0, 0)
            released+=1
        if len(self.in2)>0:
            self.container.append(self.in2[0])
            self.in2=np.delete(self.in2, 0, 0)
            released+=1
        self.update_distance()
        return released

    def update_distance(self):
        self.label = np.zeros(len(self.target))
        self.dists = []
        self.ids = []
        for i in range(len(self.container)):
            self.dists.append(self.get_distance(self.container[i]))
            self.ids.append(self.get_distance(self.container[i]))

    def remove_order(self, o):
        for i in range(len(self.target)):
            if np.array_equal(self.target[i],o):
                self.target=np.delete(self.target, i, 0)
                return None


    def release(self):
        for o in range(len(self.container)):
            for po in range(len(self.pending_list)):
                if np.array_equal(self.container[o], self.pending_list[po]):
                    d=self.dists[o]
                    d = max(0, d - 1)
                    self.remove_order(self.pending_list[po])
                    del self.container[o]
                    del self.dists[o]
                    del self.pending_list[po]
                    self.cursor -=1
                    return d
        # n = np.random.randint(len(self.dists))
        # n = np.argmax(self.dists)
        candidate=[int(self.dists[i]==0)*10000+self.dists[i] for i in range(len(self.dists))]
        n = np.argmin(candidate)
        d=self.dists[n]
        self.remove_order(self.container[n])
        del self.container[n]
        del self.dists[n]
        if not d==1:
            self.pending_list.append(self.target[self.cursor])
            self.cursor += 1
        d = max(0, d - 1)
        # self.released[]
        return d



# C : container
# P : pending list

# for o in C:
#     for po in P:
#         if o == po:
#             release(o)
#             update pending list
#             return
# # if no pending orders in container, release the order with minimal tardiness
# t=Inf
# for o in C:
#     if tard(o) <t:
#         t=tard(o)
#         r=o
# release(r)
# update pending list
# return
#
# input_sequence=get_sequence('input_test.xls')
# result_sequence1=get_sequence('result_test.xls')
# result_sequence2=get_sequence('result_test.xls', sheet=1)
root_path='data'
number_of_orders=100
capacities=[10]
# capacit ies=[350]
result=[]
input_sequence=get_sequence('input_test.xls')
# input_sequence=get_sequence('Biddinglist_Test2_input.xls')
for test_num in range(12):
    row=[]
    result.append(row)
    # file_input1 = os.path.join(root_path, 'Color_and_models1_{}_.xls'.format(test_num))
    # file_input2 = os.path.join(root_path, 'Color_and_models2_{}_.xls'.format(test_num))
    file_input1 = os.path.join('', 'result_test.xls'.format(test_num))
    file_input2 = os.path.join('', 'result_test.xls'.format(test_num))
    for capacity in capacities:
        result_sequence1=get_sequence(file_input1)
        result_sequence2=get_sequence(file_input2,sheet=1)
        total_tardiness=0
        pbs=PBS(result_sequence1, result_sequence2, input_sequence, capacity=capacity)
        number_of_orders=0
        end=False
        while not end:
            released=pbs.fill_in()
            number_of_orders+=released
            if released==0:
                break
            else:
                for _ in range(released):
                    total_tardiness+=pbs.release()
            if number_of_orders%20==0:
                print('doing order {} out of {}'.format(number_of_orders, number_of_orders))
        row.append(total_tardiness)
        print('Test {} @capacity {} Finihsed, total tard:'.format(test_num, capacity),total_tardiness)
df = pd.DataFrame(result, columns=capacities)
df.to_excel('result0626.xls')
# print('tar:\n', pbs.target)
# print('in1:\n', pbs.in1)
# print('in2:\n',pbs.in2)

# print(pbs.get_distance(pbs.target[10]))
