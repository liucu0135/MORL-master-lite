from sequence_reader import get_sequence
import numpy as np

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
        self.container.append(self.in1[0])
        # self.dists.append(self.get_distance(self.in1[0]))
        self.container.append(self.in2[0])
        # self.dists.append(self.get_distance(self.in2[0]))
        self.in1=np.delete(self.in1, 0, 0)
        self.in2=np.delete(self.in2, 0, 0)
        self.update_distance()

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

#
# input_sequence=get_sequence('input_test.xls')
# result_sequence1=get_sequence('result_test.xls')
# result_sequence2=get_sequence('result_test.xls', sheet=1)

number_of_orders=24000
# capacity*2=PBS=COS
capacity=70
input_sequence=get_sequence('Biddinglist_Test2_input.xls')[:number_of_orders*2+capacity*2]
result_sequence1=get_sequence('distribute result2.xls')[:number_of_orders+capacity]
result_sequence2=get_sequence('distribute result2.xls', sheet=1)[:number_of_orders+capacity]


total_tardiness=0
pbs=PBS(result_sequence1, result_sequence2, input_sequence, capacity=capacity)
print(pbs.dists)
for i in range(number_of_orders):
    total_tardiness+=pbs.release()
    total_tardiness+=pbs.release()
    # print(pbs.release())
    # print(pbs.release())
    pbs.fill_in()
    if i%2000==0:
        print('doing order {} out of {}'.format(i, number_of_orders))

    # print(pbs.dists,'     ', len(pbs.pending_list))
print('total average tard:',total_tardiness/number_of_orders)
print('total tard:',total_tardiness)
# print('tar:\n', pbs.target)
# print('in1:\n', pbs.in1)
# print('in2:\n',pbs.in2)

# print(pbs.get_distance(pbs.target[10]))
