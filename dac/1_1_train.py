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
exp = quadruple_tank_bias
data_file = 'res/1_1data_collect_all_pointsquadruple_tank_bias.csv'
df = pd.read_csv (data_file)
y = df['y'].to_numpy()
alarm = df['alarm_list'].to_numpy()
n = exp.model.cur_y.size
y_list = np.empty((y.size, n))
alarm_list = np.empty((y.size, 1))
index = 0
for i in y:
    i = i.replace('[', '')
    i = i.replace(']', '')
    # print(i)
    temp = i.split()
    for j in range(0, n):
        y_list[index][j] = float(temp[j])
    alarm_list[index] = bool(alarm[index])
    index += 1
