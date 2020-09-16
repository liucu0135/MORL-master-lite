#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the arrayManipulation function below.
def arrayManipulation(n, queries):
    dev=[0]*n
    for x,y,inc in queries:
        dev[x-1]+=inc
        if y<len(dev):
            dev[y]-=inc
    max=0
    temp=0
    for d in dev:
        temp=d+temp
        if max<temp:
            max=temp
    return max

if __name__ == '__main__':
    # fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    queries = []

    for _ in range(m):
        queries.append(list(map(int, input().rstrip().split())))

    result = arrayManipulation(n, queries)
    print(result)
    # fptr.write(str(result) + '\n')

    # fptr.close()
