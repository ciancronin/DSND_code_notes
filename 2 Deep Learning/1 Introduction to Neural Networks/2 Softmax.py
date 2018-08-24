#
# @author - Cian Cronin (croninc@google.com)
# @description - 2 Softmax 
# @date - 24/08/2018
#

import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    
    softmax_L = []
    
    expL = np.exp(L)
    sumExpL= np.sum(expL)

    for i in range(len(L)):
        softmax_L.append(expL[i]/sumExpL)

    return softmax_L

#softmax using list comprehensions
def softmax_lc(L):
    softmax_L = [np.exp(i)/np.sum(np.exp(L)) for i in L]

    return softmax_L


if __name__ == '__main__':
    L = [1,2,3,4,5,6,7,8,9,10]

    L = list(range(1,100))
    print(L)
    print("")
    #print(softmax(L))
    print(np.sum(softmax(L)))
    print("")
    #print(softmax_lc(L))
    print(np.sum(softmax_lc(L)))