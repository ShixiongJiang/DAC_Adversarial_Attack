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

os.environ["RANDOM_SEED"] = '0'  # for reproducibility

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


class System:
    def __init__(self, detector, exp, query_type='square', N_query=10, N_step=1, query_start_index=100, model=None): # N_step is the number of step to
        # train the detector with N query
        self.data_file = ('res/1_1data_collect_all_points' + exp.name + '.csv')
        exp_name = f"{exp.name}"
        # logger.info(f"{exp_name:=^40}")
        A = exp.model.sysd.A
        B = exp.model.sysd.B
        kf_C = exp.kf_C
        C = exp.model.sysd.C
        D = exp.model.sysd.D
        kf_Q = exp.model.p_noise_dist.sigma if exp.model.p_noise_dist is not None else np.zeros_like(A)
        kf_R = exp.kf_R
        kf_P = np.zeros_like(A)
        kf = KalmanFilter(A, B, C, D, kf_Q, kf_R)
        detector = detector

        self.query = Query(exp.y_up, exp.y_low, K=8, start_index=query_start_index, N_step=N_step)
        x_update = None
        self.index_list = []
        self.reference_list = []
        self.x_update_list = []
        self.y_list = []
        self.control_list = []
        self.alarm_list = []
        self.residual_list = []
        if model is not None:
            self.query.set_model(model)
        end_query = False
        i = 0
        step = N_step
        if query_type == 'active_learn':
            self.query.active_learn(N_query)
        while True:
            # assert exp.model.cur_index == i
            exp.model.update_current_ref(exp.ref[i])
            exp.model.evolve()
            if i == 0:
                x_update = exp.model.cur_x
            if i > 0:
                # exp.model.cur_y = exp.attack.launch(exp.model.cur_y, i, exp.model.states)

                exp.model.cur_y, end_query = self.query.launch(exp.model.cur_y, exp.model.cur_index, query_type)

                x_update, P_update, residual = kf.one_step(x_update, kf_P, exp.model.cur_u, exp.model.cur_y)
                exp.model.cur_feedback = x_update
                kf_P = P_update
                if detector.name == 'CUSUM':
                    alarm = False
                    for i in range(residual.size):
                        alarm = alarm or detector.detect(residual[i])
                else:
                    alarm = detector.detect(residual)
                # logger.debug(f"i = {exp.model.cur_index}, state={exp.model.cur_x}, update={x_update},y={exp.model.cur_y}, residual={residual}, alarm={alarm}")
                if exp.model.cur_index >= self.query.start_index:
                    self.index_list.append(exp.model.cur_index)
                    self.reference_list.append(exp.ref[i])
                    self.x_update_list.append(x_update)
                    self.y_list.append(exp.model.cur_y)
                    self.control_list.append(exp.model.cur_u)
                    self.alarm_list.append(alarm)
                    self.residual_list.append(residual)

                if exp.model.cur_index >= query_start_index and exp.model.cur_index - query_start_index >= N_step - 1:
                    # print(f"model reset at {exp.model.cur_index}")
                    exp.model.reset()
                    if end_query:
                        break
                    i = 0
            i += 1

# def write_file(System):
#
#     df = pd.DataFrame({"reference": System.reference_list, "x_update": System.x_update_list, 'y': System.y_list, 'control': System.control_list
#                               , 'alarm_list': System.alarm_list, 'residual': System.residual_list})
#     df.to_csv(System.data_file, index=True)
#     for i in range(len(System.reference_list)):
#         if not System.alarm_list[i]:
#             plt.scatter(System.y_list[i][0], System.y_list[i][1], c="blue")
#         else:
#             plt.scatter(System.y_list[i][0], System.y_list[i][1], c="red")
#     plt.xlim([0, 5])
#     plt.ylim([0, 5])
#     plt.show()
