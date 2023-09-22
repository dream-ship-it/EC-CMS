import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# M: base聚类数量
class EC_CMC_al:
    def __init__(self, alpha, lamda, r1 = 1, r2 = 1) -> None:
        self.alpha = alpha
        self.lamda = lamda
        self.r1 = r1
        self.r2 = r2

    
    def setCA(self, LWCA_matrix, CA):
        # 高度可信度的H矩阵 LWCA方法生成的CA矩阵
        self.HC = self.getHC(CA=CA)
        self.LWCA_matrix = LWCA_matrix

    def setclsNums(self, clsNums):
        self.clsNums = clsNums
    
    def getHC(self, CA):
        E = np.copy(CA)
        E[CA >= self.alpha] = 0
        A = CA - E
        return A
    
    def solute(self, H, A):
        n = A.shape[0]
        t = 0
        e = 1e-2
        max_iter = 100
        I = np.eye(n)
        C = np.zeros((n, n))
        E = C.copy()
        F = C.copy()
        r1 = self.r1
        r2 = self.r2
        Y1 = A.copy()
        Y2 = C.copy()
        D = H.sum(axis=1)
        phi = np.diag(D) - H
        inv_part = np.linalg.inv(2 * phi + (r1 + r2) * I)

        while t < max_iter:
            t = t + 1

            # update C
            Ct = C.copy()
            P1 = A - E + Y1 / r1
            P2 = F - Y2 / r2
            C = np.matmul(inv_part, r1 * P1 + r2 * P2)

            # update E
            Et = E.copy()
            E = r1 * (A - C) + Y1
            E = E / (self.lamda + r1)
            E[H > 0] = 0

            # update F
            Ft = F.copy()
            F = C + Y2 / r2
            F = np.minimum(np.maximum((F + F.T) / 2, 0), 1)

            # update Y
            Y1t = Y1.copy()
            residual1 = A - C - E
            Y1 = Y1t + r1 * residual1

            Y2t = Y2.copy()
            residual2 = C - F
            Y2 = Y2t + r2 * residual2

            diffC = np.abs(np.linalg.norm(C - Ct, 'fro') / (np.linalg.norm(Ct, 'fro') + 1e-6))
            diffE = np.abs(np.linalg.norm(E - Et, 'fro') / (np.linalg.norm(Et, 'fro') + 1e-6))
            diffF = np.abs(np.linalg.norm(F - Ft, 'fro') / (np.linalg.norm(Ft, 'fro') + 1e-6))
            diffY1 = np.abs(np.linalg.norm(residual1, 'fro') / (np.linalg.norm(Y1t, 'fro') + 1e-6))
            diffY2 = np.abs(np.linalg.norm(residual2, 'fro') / (np.linalg.norm(Y2t, 'fro') + + 1e-6))

            if max([diffC, diffE, diffF, diffY1, diffY2]) < e:
                break

        return C, E, t

    
    def run(self):
        # 对角元素置零
        N = self.LWCA_matrix.shape[0] # 样本数量
        HC = self.HC - np.diag(np.diag(self.HC))
        C, _E, _ = self.solute(HC, self.LWCA_matrix)
        # 使得C为对称矩阵
        C = np.triu(C)
        C += C.T - np.diag(C.diagonal())
        s = squareform(C - np.diag(np.diag(C)), 'tovector')
        d = 1 - s
        results = fcluster(linkage(d, 'average'), self.clsNums, criterion='maxclust')
        return results