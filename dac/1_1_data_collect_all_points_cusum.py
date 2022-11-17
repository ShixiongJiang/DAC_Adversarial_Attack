from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import logging
import sys
import os
from time import perf_counter
import csv
import pandas as pd
from utils.query import Query
import matplotlib.pyplot as plt
os.environ["RANDOM_SEED"] = '0'   # for reproducibility

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
# exps = [motor_speed_bias]
# exps = [rlc_circuit_bias]
exps = [quadruple_tank_bias]
colors = {'none': 'red', 'lp': 'cyan', 'lqr': 'blue', 'ssr': 'orange', 'oprp': 'violet', 'oprp-open': 'purple'}
result = {}  # for print or plot

# logger
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)




for exp in exps:
    data_file = 'res/1_1data_collect_all_points_cusum_'+exp.name+'.csv'
    # headers = ['']
    # times = 50
    # with open(data_file, 'w', newline='') as f:
    #     writer = csv.writer(f)
        # writer.writerow(headers)
    exp_name = f"{exp.name} "
    logger.info(f"{exp_name:=^40}")
    A = exp.model.sysd.A
    B = exp.model.sysd.B
    kf_C = exp.kf_C
    C = exp.model.sysd.C
    D = exp.model.sysd.D
    kf_Q = exp.model.p_noise_dist.sigma if exp.model.p_noise_dist is not None else np.zeros_like(A)
    kf_R = exp.kf_R
    kf_P = np.zeros_like(A)
    kf = KalmanFilter(A, B, C, D, kf_Q, kf_R)
    C_filter = np.array([1, 0])
    detector = CUSUM(drift=0.02, threshold=0.3)
    # detector = chi_square(threshold=4.61)
    x_update = None
    reference_list = []
    x_update_list = []
    y_list = []
    control_list = []
    alarm_list = []
    residual_list = []
    end_query = False
    i = 0
    # y_low = np.array([0])
    # y_up = np.array([10])
    # query_start_index = 200
    # exp.query = Query(y_up, y_low, K=64, start_index=query_start_index)
    while not end_query:
        # assert exp.model.cur_index == i
        exp.model.update_current_ref(exp.ref[i])
        exp.model.evolve()
        if i == 0:
            x_update = exp.model.cur_x
        if i > 0:
            # exp.model.cur_y = exp.attack.launch(exp.model.cur_y, i, exp.model.states)
            exp.model.cur_y, end_query = exp.query.launch(exp.model.cur_y, exp.model.cur_index)
            x_update, P_update, residual = kf.one_step(x_update, kf_P, exp.model.cur_u, exp.model.cur_y)
            exp.model.cur_feedback = x_update
            kf_P = P_update
            alarm_1 = detector.detect(residual[0])
            alarm_2 = detector.detect(residual[1])
            alarm = alarm_1 or alarm_2
            # logger.debug(f"i = {exp.model.cur_index}, state={exp.model.cur_x}, update={x_update},y={exp.model.cur_y}, residual={residual}, alarm={alarm}")
            if exp.model.cur_index >= exp.query.start_index:
                reference_list.append(exp.ref[i])
                x_update_list.append(x_update)
                y_list.append(exp.model.cur_y)
                control_list.append(exp.model.cur_u)
                alarm_list.append(alarm)
                residual_list.append(residual)

            if exp.model.cur_index >= exp.query_start_index:
                exp.model.reset()
                i = 0
        i += 1

    df = pd.DataFrame({"reference": reference_list, "x_update": x_update_list, 'y':y_list, 'control':control_list
                       , 'alarm_list': alarm_list, 'residual':residual_list})
    df.to_csv(data_file, index=True)
    for i in range(len(reference_list)):
        if not alarm_list[i]:
            plt.scatter(y_list[i][0], y_list[i][1], c="blue")
        else:
            plt.scatter(y_list[i][0], y_list[i][1], c="red")
    plt.xlim([0, 5])
    plt.ylim([0, 5])
    plt.show()

    # print(index)
    # print(len(alarm_list))