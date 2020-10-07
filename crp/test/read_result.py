import pandas as pd
import struct
import numpy as np
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

def read_result(f):
    a,b=np.loadtxt(f,dtype=np.float, delimiter=',',usecols=(1,2),unpack=True)
    return a.tolist(),b.tolist()




