import numpy as np
from numpy.linalg import multi_dot, norm
import random
from scipy.optimize import minimize
import time
from matplotlib import pyplot as plt
from collections import Counter
import functools
import multiprocessing as mp 
import math

n = int(input("Enter the number of iterations: "))
# n = 20

start = time.time()

p = []

Parameter = [0.018395153137267837, 0.024684807777732274,
            0.029707767026692415, 0.030079353579520685,
            0.00651245126325279, 0.02949553743427049,
            0.01159096408344912, 0.010829363907230143,
            0.010866882734424523, 0.023839542568108416,
            0.0010384663860012143, 0.009624641668396019,
            0.02907918891592665, 0.009774251569554363,
            0.019901159456116375, 0.011240837071685312,
            0.0032511141684279502, 0.005437955411988602,
            0.006409454901653719, 0.01878097144345906,
            0.030726183785252912, 0.030700307765529147,
            0.017034486252652012, 0.020454826725749654,
            0.14050653880210734, 0.09313983447651784]


#Gate initialisation (H,CX,ID,X,Y,Z,S,T)
hg = np.array([[1,0,0,0],[0,0,0,1],[0,0,-1,0],[0,1,0,0]])
xg = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]])
yg = np.array([[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,-1]])
zg = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
iden = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
cx10 = np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]])
cx01 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
sg = np.array([[0,0,0,1],[0,1j,0,0],[0,0,1j,0],[1,0,0,0]])
tg = np.array([[1,0,0,0],[0,(1/np.sqrt(2)),-(1/np.sqrt(2)),0],[0,(1/np.sqrt(2)),(1/np.sqrt(2)),0],[0,0,0,1]])

#Pauli Matrices
im = np.array([[1,0],[0,1]])
xm = np.array([[0,1],[1,0]])
ym = np.array([[0,-1j],[1j,0]])
zm = np.array([[1,0],[0,-1]])

pauli1 = np.array([im,xm,ym,zm])
pauli2 = [np.kron(x,y) for x in pauli1 for y in pauli1]
all_gates = [iden,xg,yg,zg,hg,tg]

# 0 and 1 representation
z0 = [1/np.sqrt(2),0,0,1/np.sqrt(2)]
z1 = [1/np.sqrt(2),0,0,-1/np.sqrt(2)]

def multi_kron(x):
    x = np.array(x)
    return functools.reduce(np.kron,x)

#Noise model
def AD(x):
    y = np.sin(x)
    AD = np.array([[1,0,0,0],[0,np.sqrt(1-y**2),0,0],[0,0,np.sqrt(1-y**2),0],
              [y**2,0,0,1-y**2]])
    return AD

def depc(x):
    y = (np.sin(x))**2
    depc = np.array([[1,0,0,0],[0,1-4*y/3,0,0],[0,0,1-4*y/3,0],[0,0,0,1-4*y/3]])
    return depc

def bit_flip(x):
    y = np.cos(2*x)
    bit_flip = np.array([[1,0,0,0],[0,1,0,0],[0,0,y,0],[0,0,0,y]])
    return bit_flip

def pauli(para):
    q = np.ones(len(para)+1)
    dim = 2**1
    q[0] = ((math.cos(para[0]))**2)*((math.cos(para[1]))**2)*((math.cos(para[2]))**2)
    q[1] = (math.sin(para[0]))**2
    q[2] = ((math.cos(para[0]))**2)*((math.sin(para[1]))**2)
    q[3] = ((math.cos(para[0]))**2)*((math.cos(para[1]))**2)*((math.sin(para[2]))**2)
            
    sop = [[1,0,0,0],[0,q[0]+q[1]-q[2]-q[3],0,0],[0,0,q[0]+q[2]-q[1]-q[3],0],
            [0,0,0,q[0]+q[3]-q[1]-q[2]]]

    return np.matrix(sop)

def trace_distance(x,y):
    x = np.array(x)
    y = np.array(y)
    z = x-y
    return 1/2*np.sum(np.absolute(z))


def gate_matrix1(gate, n):
    if gate == 0:
        return multi_dot([AD(n[0]),pauli(n[1:4]),iden])
    elif gate == 1:
        return multi_dot([AD(n[4]),pauli(n[5:8]),tg])
    elif gate == 2:
        return multi_dot([AD(n[8]),pauli(n[9:12]),zg])
    elif gate == 3:
        return multi_dot([AD(n[12]),pauli(n[13:16]),hg])
    elif gate == 4:
        return multi_dot([AD(n[16]),pauli(n[17:20]),xg])
    elif gate == 5:
        return multi_dot([AD(n[20]),pauli(n[21:24]),yg])

def compute_oper(qub_1, noise):
    total_oper = np.identity(4)
    for i in qub_1:
        total_oper = gate_matrix1(i,noise[0:24])@total_oper
    return total_oper

def simulation_pau(x, list_gates): 
    read0 = multi_dot([bit_flip(x[24]),AD(x[25])])
    tot_oper = multi_dot([read0,compute_oper(list_gates,x)])
    final = np.dot(np.array(tot_oper),z0)
    simulation_p=[]
    outcome=[z0,z1]
    for m0 in outcome:
        simulation_p.append(np.real(np.vdot(m0,final)))
    return np.array(simulation_p)

def distance(x, counts_device, list_circuits):
    dis = 0
    for i in range(len(circ_info)):
        simulation_p = simulation_pau(x,list_circuits[i])
        device_p = np.array((counts_device[i][0]/16384,counts_device[i][1]/16384))
        dis += np.dot(simulation_p - device_p,simulation_p - device_p)
    return dis/(len(list_circuits))


def para_pau(x0, counts_device, circ_info, method):
    res = minimize(distance, x0, method=method, args=(counts_device,circ_info),
                   options={'maxiter':1000, 'disp':True, 'ftol':1.0e-10})
    return res.fun, res.x

def circuit_size(n,m):
    length_cir = n
    circ_info = {}
    for j in range(m):
        l0 = [random.randrange(0,6) for i in range(length_cir)]
        circ_info[j] = l0
    return circ_info

def counts_device(x):
    circ_prob = []
    for i in range(len(x)):
        population = [0, 1]
        weights = simulation_pau(Parameter, x[i])
        samples = random.choices(population, weights, k = 16384)
        freq = dict(Counter(samples))
        circ_prob.append(freq)
    return circ_prob


# def find_parameters(x):
#     para=para_pau(x[0],counts_device=x[2],circ_info=x[1],method='SLSQP')
#     return para

# if __name__ == "__main__": 
#     arg_set = []
#     for i in range(n):
#         circ_info = circuit_size(5,75)
#         dev_prob = counts_device(circ_info)
#         arg_set.append([[random.uniform(0,0.01*math.pi) for j in range(24)]+[random.uniform(0,0.1*math.pi) for j in range(2)],
#                         circ_info, dev_prob])
#     pool = mp.Pool(processes=3)
#     results = pool.imap_unordered(find_parameters, arg_set)
#     opt_para = list(results)
#     for t in opt_para:
#         for t1 in t.tolist():
#             p.append(t1)

for run in range(50):
    start1 = time.time()
    circ_info = circuit_size(2,150)
    dev_prob = counts_device(circ_info)

    def find_parameters(x):
        para=para_pau(x,counts_device=dev_prob,circ_info=circ_info,method='SLSQP')
        return para

    if __name__ == "__main__": 
        arg_set = []
        for i in range(n):
            arg_set.append([random.uniform(0,0.01*math.pi) for j in range(24)]+[random.uniform(0,0.1*math.pi) for j in range(2)])
        pool = mp.Pool(processes=3)
        results = pool.imap_unordered(find_parameters, arg_set)
        opt_para = list(results)
        cmp = 1
        temp = []
        for f in opt_para:
            if f[0] < cmp:
                cmp = f[0]
                temp = f[1]
        for t in temp.tolist():
            p.append(t)
    end1 = time.time()
    print(end1-start1)


with open('data_150_16_2.txt', 'w') as f:
    for item in p:
        f.write("{}\n".format(item))

end = time.time()
print(end-start)