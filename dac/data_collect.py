from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import logging
import sys
import os
from time import perf_counter
import csv
import pandas as pd

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
exps = [motor_speed_bias]
# exps = [rlc_circuit_bias]
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
    data_file = 'res/data_collect_'+exp.name+'.csv'
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
    # detector = CUSUM()
    detector = chi_square(threshold=4.61)
    x_update = None
    reference_list = []
    x_update_list = []
    y_list = []
    control_list = []
    alarm_list = []
    residual_list = []
    for i in range(0, exp.max_index + 1):
        assert exp.model.cur_index == i
        exp.model.update_current_ref(exp.ref[i])

        exp.model.evolve()
        if i == 0:
            x_update = exp.model.cur_x
        if i > 0:
            # exp.model.cur_y = exp.attack.launch(exp.model.cur_y, i, exp.model.states)
            x_update, P_update, residual = kf.one_step(x_update, kf_P, exp.model.cur_u, exp.model.cur_y)
            exp.model.cur_feedback = x_update
            kf_P = P_update
            alarm = detector.detect(residual)
            logger.debug(f"i = {i}, state={exp.model.cur_x}, update={x_update},y={exp.model.cur_y}, residual={residual}, alarm={alarm}")
            reference_list.append(exp.ref[i])
            x_update_list.append(x_update)
            y_list.append(exp.model.cur_y)
            control_list.append(exp.model.cur_u)
            alarm_list.append(alarm)
            residual_list.append(residual)

    df = pd.DataFrame({"reference": reference_list, "x_update": x_update_list, 'y':y_list, 'control':control_list
                       , 'alarm_list': alarm_list, 'residual':residual_list})
    df.to_csv(data_file, index=True)
    index = 0
    for x in alarm_list:
        if x:
            index += 1

    print(index)
    print(len(alarm_list))