#
# @author - Cian Cronin (croninc@google.com)
# @description - 3 Cross Entropy
# @date - 25/08/2018
#

import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    p_log = [np.log(p) for p in P]
    p_log_minus = [np.log(1 - p) for p in P]

    cross_entropy = 0.0

    for i in range(len(Y)):
    	cross_entropy += -1 *  ((Y[i]*p_log[i]) + ((1 - Y[i]) * p_log_minus[i]))

    return cross_entropy

def cross_entropy_sol(Y, P):
	Y = np.float_(Y)
	P = np.float_(P)
	return -np.sum((Y * np.log(P)) + ((1 - Y)* np.log(1 - P)))


if __name__ == '__main__':
	Y = [1, 1, 0]
	P = [0.8, 0.7, 0.1]

	print(cross_entropy(Y, P))
	print(cross_entropy_sol(Y, P))