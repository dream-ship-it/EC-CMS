import scipy.io as scio
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import f1_score
from LWCA import computeECI, computeLWCA
from util import get_all_segs, getCA
from EC_CMS import EC_CMC_al
from solver import solver

rs = [0.01, 0.1, 1, 10, 100]
for r1 in rs:
    al = EC_CMC_al(alpha = 0.8, lamda = 0.4, r1=r1)
    s1 = solver(dataName='UMIST', nums = 20, seed = 2, cntTimes = 20, algorithm = al)
    print("γ1为{0}时，ari".format(r1, s1.ari))

for r2 in rs:
    al = EC_CMC_al(alpha = 0.8, lamda = 0.4, r2=r2)
    s1 = solver(dataName='UMIST', nums = 20, seed = 2, cntTimes = 20, algorithm = al)
    print("γ2为{0}时，ari".format(r2, s1.ari))