import pandas
import struct
import numpy
import matplotlib.pyplot as plt
import copy

def find_pareto(data):
    prt=[]
    # set=numpy.split(set, axis=0)
    for i in range(data.shape[0]):
        dominated=False
        s=data[i,:]
        if len(prt)==0:
            prt.append(s)
            dominated=True
        for j in range(len(prt)):
            if prt[j][0]<=s[0] and prt[j][1]<=s[1]:
                prt[j]=s
                dominated=True
                break
        if not dominated:
            prt.append(s)
    prt=numpy.stack(prt,axis=0)
    if len(prt)<data.shape[0]:
        prt=find_pareto(prt)
    return prt

def read_result():
    f=open('./test/scalerized_test3.txt','r')
    lines=f.readlines()
    scale=1
    scales=[]
    record = numpy.zeros((98, 2))
    for l in lines:
        if 'scale' in l:
            scale+=1
            scales.append(record)
            record = numpy.zeros((98, 2))
            continue
        if 'episode' not in l:
            continue
        nums=l.split(', ')
        e=int(nums[0][8:10])-2
        nc=float(nums[5][3:])
        cc=float(nums[6][3:])
        record[e,0]=nc
        record[e,1]=cc
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)

    # scales=[-find_pareto(-s) for s in scales]
    #
    # for s in scales:
    #     ax1.scatter(s[:,0], s[:,1])
    scales=numpy.concatenate(scales, axis=0)
    return scales

