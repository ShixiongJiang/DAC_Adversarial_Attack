# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7322210&tag=1
import numpy as np
from numpy.linalg import inv


class chi_square():
    def __init__(self, threshold=500):
        self.threshold = threshold

    def detect(self, z_t):
        z_t = np.reshape(z_t, (z_t.size, 1))
        # print(z_t.shape)
        p_k = np.cov(z_t, rowvar=False)
        p_k = np.array([(p_k)])
        # print(p_k)

        # print(z_t.T.shape)
        g_t = z_t.T * inv(p_k.reshape(1,1)) @ z_t
        print(g_t)
        if g_t > self.threshold:
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

