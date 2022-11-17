import numpy as np
import pandas as pd

from settings_baseline import motor_speed_bias, quadruple_tank_bias, lane_keeping, f16_bias, aircraft_pitch_bias, \
    boeing747_bias, platoon_bias, quadrotor_bias, rlc_circuit_bias
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
from system import System
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def show_boundary(model, y_list):
    ans = model.predict(y_list)

    for i in range(len(y_list)):
        if not ans[i]:
            plt.scatter(y_list[i][0], y_list[i][1], c="blue", s=3)
        else:
            plt.scatter(y_list[i][0], y_list[i][1], c="red", s=3)
    plt.xlim([0, 5])
    plt.ylim([0, 5])
    plt.show()



def read_file(filename):
    data_file = 'res/' + filename
    df = pd.read_csv(data_file)
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
    return y_list, alarm_list


def write_file(filename, sys):
    df = pd.DataFrame(
        {"reference": sys.reference_list, "x_update": sys.x_update_list, 'y': sys.y_list, 'control': sys.control_list
            , 'alarm_list': sys.alarm_list, 'residual': sys.residual_list})
    # data_file = 'res/1_1data_collect_all_points_cusum_quadruple_tank_bias.csv'
    data_file = 'res/' + filename
    df.to_csv(data_file, index=True)


detector = CUSUM(drift=0.02, threshold=0.3)
exp = quadruple_tank_bias
query_type = 'square'
sys = System(detector=detector, exp=exp, query_type=query_type)
filename = detector.name + "_" + exp.name
write_file(filename=filename, sys=sys)

# data_file = 'res/1_1data_collect_all_pointsquadruple_tank_bias.csv'

# for i in range(len(y_list)):
#     if not alarm[i]:
#         plt.scatter(y_list[i][0], y_list[i][1], c="blue")
#     else:
#         plt.scatter(y_list[i][0], y_list[i][1], c="red")
# plt.xlim([0, 5])
# plt.ylim([0, 5])
# plt.show()


# data_file = 'res/1_1data_collect_all_points_cusum_quadruple_tank_bias.csv'
y_list, alarm_list = read_file(filename)
# print(alarm_list.size)
knn = KNeighborsClassifier()
knn.fit(y_list, alarm_list)
# print(logisticRegr.get_params(True))
show_boundary(knn, y_list)

# print(ans)
exp.query.set_model(knn)

active_itr = 5
for i in range(active_itr):
    exp = quadruple_tank_bias
    sys = System(detector=detector, exp=exp, query_type='active_learn',N_query=20)
    write_file(filename=filename, sys=sys)
    y_list_1, alarm_list_1 = read_file(filename)
    y_list = np.concatenate((y_list, y_list_1))
    alarm_list = np.concatenate((alarm_list, alarm_list_1))
    knn.fit(y_list, alarm_list)

show_boundary(knn, y_list)
# print(xb1)