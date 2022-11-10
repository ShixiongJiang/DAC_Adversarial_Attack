import numpy as np
import pandas as pd

from settings_baseline import motor_speed_bias, quadruple_tank_bias, lane_keeping, f16_bias, aircraft_pitch_bias, boeing747_bias, platoon_bias, quadrotor_bias, rlc_circuit_bias
from utils.formal.gaussian_distribution import GaussianDistribution
from utils.formal.reachability import ReachableSet
from utils.formal.zonotope import Zonotope
from utils.observers.kalman_filter import KalmanFilter
from utils.observers.full_state_bound import Estimator
from utils.controllers.LP_cvxpy import LP
from utils.controllers.MPC_cvxpy import MPC
from utils.detector.cusum import CUSUM
from utils.detector.chi_square import chi_square
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

exp = quadruple_tank_bias
data_file = 'res/1_1data_collect_all_pointsquadruple_tank_bias.csv'
df = pd.read_csv (data_file)
y = df['y'].to_numpy()
alarm = df['alarm_list'].to_numpy()
ref = df['reference'].to_numpy()
n = exp.model.cur_y.size
y_list = np.empty((y.size, n))
alarm_list = np.empty((y.size))
ref_list = np.empty((y.size, 1))
index = 0
for i in y:
    i = i.replace('[', '')
    i = i.replace(']', '')
    # print(i)
    temp = i.split()

    for j in range(0, n):
        y_list[index][j] = float(temp[j])
        # print(y_list[index])
    alarm_list[index] = bool(alarm[index])

    # alarm_list[index] = not bool(alarm[index])
    index += 1



for i in range(len(y_list)):
    if not alarm[i]:
        plt.scatter(y_list[i][0], y_list[i][1], c="blue")
    else:
        plt.scatter(y_list[i][0], y_list[i][1], c="red")
plt.xlim([0, 5])
plt.ylim([0, 5])
plt.show()

knn = KNeighborsClassifier()
knn.fit(y_list, alarm_list)
# print(logisticRegr.get_params(True))
ans = knn.predict(y_list)
# for i in ans:
#     if i == 0:
#         print(i)
# for i in range(len(y_list)):
#     if not ans[i]:
#         plt.scatter(y_list[i][0], y_list[i][1], c="blue")
#     else:
#         plt.scatter(y_list[i][0], y_list[i][1], c="red")
# plt.xlim([0, 5])
# plt.ylim([0, 5])
# plt.show()
# print(ans)
