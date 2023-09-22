import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

def getOneClsEntropy(partBcs):
    # 获取一个聚类对于集成的熵值
    # 聚类的总熵值通过计算其在所有基础聚类中的熵值之和得到。
    E = 0
    for i in range(partBcs.shape[1]):
        tmp = np.sort(partBcs[:, i])  # 对当前基础聚类分段进行排序
        uTmp = np.unique(tmp)  # 获取排序后的唯一值

        if len(uTmp) <= 1:  # 如果唯一值只有一个，则跳过计算熵值的过程
            continue

        cnts = np.zeros(len(uTmp))  # 初始化每个唯一值的计数数组
        for j in range(len(uTmp)):
            cnts[j] = np.sum(tmp == uTmp[j])  # 统计每个唯一值的出现次数

        cnts = cnts / np.sum(cnts)  # 将计数数组归一化
        E = E - np.sum(cnts * np.log2(cnts))  # 计算熵值

    return E

def getAllClsEntropy(bcs, baseClsSegs):
    baseClsSegs = baseClsSegs.T  # 转置基础聚类分段数组
    _, nCls = baseClsSegs.shape  # 获取聚类数目
    Es = np.zeros(nCls)  # 初始化每个聚类的熵值数组
    for i in range(nCls):
        partBcs = bcs[baseClsSegs[:, i] != 0, :]  # 提取与当前聚类对应的部分基础聚类分段
        Es[i] = getOneClsEntropy(partBcs)  # 计算当前聚类对于集成的熵值
    return Es

def computeECI(bcs, baseClsSegs, para_theta):
    M = bcs.shape[1]  # 获取基础聚类数目 M
    ETs = getAllClsEntropy(bcs, baseClsSegs)  # 获取每个聚类对于集成的熵值
    ECI = np.exp(-ETs / para_theta / M)  # 计算每个聚类的集成一致性指数 ECI
    return ECI 

def computeLWCA(base_cls_segs, ECI, M):
    # 转置 base_cls_segs 矩阵
    base_cls_segs = base_cls_segs.T
    N = base_cls_segs.shape[0]
    
    # 使用广播将 base_cls_segs 的每一列与 ECI 的对应元素相乘，
    # 然后将结果与转置的 base_cls_segs 矩阵相乘并除以 M
    LWCA_matrix = np.dot(np.multiply(base_cls_segs, ECI[:, np.newaxis].T), base_cls_segs.T) / M
    
    # 将 LWCA 的对角线元素设置为零，并加上一个单位矩阵
    np.fill_diagonal(LWCA_matrix, 0)
    LWCA_matrix += np.eye(N)
    return LWCA_matrix

class LWCA:
    def __init__(self) -> None:
        pass
    
    def setCA(self, LWCA_matrix, CA):
        self.LWCA_matrix = LWCA_matrix

    def setclsNums(self, clsNums):
        self.clsNums = clsNums

    def run(self):
        N = self.LWCA_matrix.shape[0]
        def stod2(S):
            N = S.shape[0]
            s = np.zeros(N * (N - 1) // 2)
            nextIdx = 0
            for a in range(N - 1):  # Change matrix's format to be input of linkage function
                s[nextIdx:nextIdx + N - a - 1] = S[a, a + 1:]
                nextIdx += N - a - 1
            d = 1 - s  # Compute distance (d = 1 - sim)
            return d
        d = stod2(self.LWCA_matrix)
        results = fcluster(linkage(d, method='average'), self.clsNums, criterion='maxclust')
        return results
