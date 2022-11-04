# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7322210&tag=1
import numpy as np
from numpy.linalg import inv

"""
for any column vectors, the covariance matrix
    Cov(x) = E[ZZT] - (E[Z])(E[Z])T
"""

class chi_square():
    def __init__(self, threshold=0.1):
        self.threshold = threshold
        self.Z_k = None
    # def columnCon(self, z):
    #     cov =

    def detect(self, z_k):
        if self.Z_k is None:
            self.Z_k = z_k
        else:
            self.Z_k = np.vstack((self.Z_k, z_k))
      
        # z_k = np.reshape(z_k, (z_k.size, 1))
        # print(z_t.shape)
        P_k = np.cov(self.Z_k, rowvar=False)
        P_k = np.array([(P_k)])
        # print(p_k)

        # print(z_t.T.shape)
        g_k = self.Z_k.T * inv(P_k.reshape(1,1)) @ self.Z_k
        print(g_k)
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

