import numpy as np
from scipy.sparse import lil_matrix

def get_all_segs(base_cls):
    """
    获取所有基础聚类的类别分段信息

    参数：
    base_cls: numpy 数组，形状为 (N, M)，N 为数据点的数量，M 为基础聚类的数量

    返回值：
    bcs: numpy 数组，形状为 (N, M)，存储每个基础聚类的类别分段信息
    base_cls_segs: 稀疏矩阵，形状为 (n_cls, N)，存储所有基础聚类中的类别信息
    """

    N, M = base_cls.shape
    # N：数据点的数量
    # M：基础聚类的数量
    bcs = np.copy(base_cls)
    # 计算每个基础聚类中的最大类别编号
    n_cls_orig = np.max(bcs, axis=0)
    # 计算每个基础聚类中类别编号的累积和
    C = np.cumsum(n_cls_orig)
    # 对 bcs 中的每列进行偏移，以避免不同基础聚类的类别编号冲突
    bcs += np.hstack(([0], C[:-1]))
    # 计算所有基础聚类中总共的类别数
    n_cls = n_cls_orig[-1] + C[-1]
    # 创建稀疏矩阵存储基础聚类的类别信息
    base_cls_segs = np.zeros((n_cls, N), dtype=int)
    # 将基础聚类的类别信息存储在稀疏矩阵中
    for i in range(M):
        base_cls_segs[bcs[:, i], np.arange(N)] = 1

    return bcs, base_cls_segs

# 将baseClsSegs转为CA矩阵
def getCA(baseClsSegs, M):
    CA = np.dot(baseClsSegs.T, baseClsSegs) / M
    return CA

def compute_f(T, H):
    if len(T) != len(H):
        print("Size of T:", len(T))
        print("Size of H:", len(H))
    
    N = len(T)
    numT = 0
    numH = 0
    numI = 0
    for n in range(N):
        Tn = (T[n+1:] == T[n])
        Hn = (H[n+1:] == H[n])
        numT += sum(Tn)
        numH += sum(Hn)
        numI += sum(Tn * Hn)
    
    p = 1
    r = 1
    f = 1
    if numH > 0:
        p = numI / numH
    if numT > 0:
        r = numI / numT
    if (p + r) == 0:
        f = 0
    else:
        f = 2 * p * r / (p + r)
    
    return f
