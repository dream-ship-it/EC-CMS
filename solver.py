import scipy.io as scio
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import f1_score
from LWCA import computeECI, computeLWCA
from util import get_all_segs, getCA, compute_f

# 求解器
class solver:
    # 初始化
    # dataName：使用数据集； cntTimes：运行次数；nums：base聚类数量， seed随机种子； algorithm：使用算法
    def __init__(self, dataName, cntTimes, nums, seed, algorithm) -> None:
        self.cntTimes = cntTimes
        self.seed = seed
        data = scio.loadmat('dataset/{0}.mat'.format(dataName))
        self.dataName = dataName
        self.M = nums
        self.gt = np.ravel(data['gt'])
        self.clsNums = len(np.unique(self.gt))
        self.members = data['members']
        self.algorithm = algorithm

    def run(self):
        np.random.seed(seed = self.seed)
        N, poolSize = self.members.shape
        NMI = np.zeros(self.cntTimes) # 存NMI
        ARI = np.zeros(self.cntTimes) # 存ARI
        F = np.zeros(self.cntTimes) # 存F
        # 生成一个随机选择
        # bcIdx = np.zeros((self.cntTimes, self.M), dtype=int)
        # for i in range(self.cntTimes):
        #     tmp = np.random.permutation(poolSize)
        #     bcIdx[i, :] = tmp[: self.M]
        for runidx in range(self.cntTimes):
            baseCls = self.members[:, np.random.choice(poolSize, self.M, replace = False)].astype(int)
            bcs, baseClsSegs = get_all_segs(baseCls)
            # 计算ECI, LWCA
            ECI = computeECI(bcs = bcs, baseClsSegs = baseClsSegs, para_theta = 0.4)
            LWCA_matrix = computeLWCA(baseClsSegs, ECI, self.M)
            # 执行模型
            CA = getCA(baseClsSegs=baseClsSegs, M = self.M)
            self.algorithm.setCA(LWCA_matrix = LWCA_matrix, CA = CA)
            self.algorithm.setclsNums(clsNums = self.clsNums)
            results = self.algorithm.run()
            if np.min(results) == 0:
                results += 1
            NMI[runidx] = normalized_mutual_info_score(results, self.gt)
            ARI[runidx] = adjusted_rand_score(results, self.gt)
            F[runidx] = compute_f(results, self.gt)
        # 结果
        nmi = np.mean(NMI)
        varnmi = np.std(NMI)
        ari = np.mean(ARI)
        varari = np.std(ARI)
        f = np.mean(F)
        varf = np.std(F)
        # 展示结果
        print('**************************************************************')
        print('** Average Performance over', self.cntTimes, 'runs on the', self.dataName, 'dataset **')
        print('Data size:', N)
        print('Ensemble size:', self.M)
        print('Average NMI/ARI/F scores:')
        print('EC_CMS :', nmi, ari, f)
        print('**************************************************************')
        print('**************************************************************')
        self.nmi = nmi
        self.ari = ari
        self.f = f
        