
import random
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt 

Q = {}  # item
P = {}  # user
K = 20 # feature size
REG_PARAM = 0.12  # regularization parameter
ITER = 40  # number of iteration
L_RATE = 0.000001 # learning rate
TRAIN_FILE = "12month_train.txt"

def random_k_array():
    rand_list = []
    for i in range(K):
        rand_list.append(random.uniform(0.0, (100.0 / K) ** .5))
    return np.asarray(rand_list)

def initialization():
    with open(TRAIN_FILE, mode='r') as f:
        line = f.readline()
        
        while line:
            uid, mid, _ = line.split()
            if uid not in P:
                P[uid] = random_k_array()
            if mid not in Q:
                Q[mid] = random_k_array()
            line = f.readline()


def sgd():
    initialization()
    error_list = []
    iter_list = []
    for num in range(ITER):
        with open(TRAIN_FILE, mode='r') as f:
            line = f.readline()
            while line:
                u, i, r_iu = line.split()
                r_iu = float(r_iu)
                old_q_i = Q[i]
                old_p_u = P[u]
                e_iu = 2.0 * (r_iu - np.dot(old_q_i, old_p_u))
                q_i = old_q_i + L_RATE * (e_iu * old_p_u - 2 * REG_PARAM * old_q_i)
                p_u = old_p_u + L_RATE * (e_iu * old_q_i - 2 * REG_PARAM * old_p_u)
                Q[i] = q_i
                P[u] = p_u
                line = f.readline()
            iter_list.append(num + 1)
            cur_error = compute_error()
            error_list.append(compute_error())
            print(error_list)
    print(error_list)
    gen_predict()
    # plt.xlabel('Iteration')
    # plt.ylabel('Error')
    # plt.scatter(iter_list , error_list, alpha=0.5)
    # plt.savefig("error.png", bbox_inches="tight")
def gen_predict():
    f_out = open("predict.txt", "w")
    for u in P:
        for i in Q:
            f_out.write(u + "\t" + i + '\t')
            f_out.write(str(np.dot(Q[i], P[u])) + "\n")
    f_out.close()

def compute_error():
    error = 0
    with open(TRAIN_FILE, mode='r') as f:
        line = f.readline()
        while line:
            u, i, r_iu = line.split()
            r_iu = float(r_iu)
            error += (r_iu - np.dot(Q[i], P[u])) ** 2
            line = f.readline()
    for i in Q:
        error += REG_PARAM * (LA.norm(Q[i], ord=2) **2 )
    for u in P:
        error += REG_PARAM * (LA.norm(P[u], ord=2) **2 )
    return error

if __name__ == "__main__":
    sgd()
