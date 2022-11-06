# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7322210&tag=1
import numpy as np
from numpy.linalg import inv

"""
for any column vectors, the covariance matrix
    Cov(x) = E[ZZT] - (E[Z])(E[Z])T
"""

class chi_square():
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.Z_k = None
    # def columnCon(self, z):
    #     cov =

    def detect(self, z_k):
        z_k = z_k[:, np.newaxis]
        if self.Z_k is None:
            self.Z_k = z_k
            # self.Z_k = self.Z_k[:, np.newaxis]
        else:
            self.Z_k = np.append(self.Z_k, z_k,  axis=1)
        if self.Z_k.size == 1:
            return False


        # z_k = np.reshape(z_k, (z_k.size, 1))
        # print(z_t.shape)
        P_k = np.cov(self.Z_k, rowvar=True, bias=True)
        P_k = np.reshape(P_k, (-1, 1))
        # print(p_k)
        inv_ma = inv(P_k)
        # print(z_t.T.shape)
        g_k = z_k.T @ inv_ma @ z_k



        if g_k > self.threshold:
            return True
        else:
            return False

if __name__ == '__main__':
    x = np.ones([3, 300]) * 10
    detector = chi_square()
    x_hat = np.ones([3, 300]) *10 + np.random.rand(3, 300) / 0.01
    # print(x_hat)
    for i in range(0, x[0].size):
        z_t = x_hat[:, i] - x[:, i]
        # print(z_t)
        # z_t.reshape(3, 1)
        alarm = detector.detect((z_t))
        if alarm:
            print(f"raise alarm at {i}")
            # break

