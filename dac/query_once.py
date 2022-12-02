import numpy as np
from settings_baseline import motor_speed_bias, quadruple_tank_bias, lane_keeping, f16_bias, aircraft_pitch_bias, boeing747_bias, platoon_bias, quadrotor_bias, rlc_circuit_bias
from utils.observers.kalman_filter import KalmanFilter
from utils.detector.chi_square import chi_square
from utils.detector.cusum import CUSUM
import queue
import joblib

# input attack_y
# output query result, y, ref
class QueryOnce:
    def __init__(self, detector, start_index=200, exp=quadruple_tank_bias, step_length=10):
        self.start_index = start_index
        self.exp = exp
        self.detector = detector
        self.end_query = False
        self.kf_Q = exp.model.p_noise_dist.sigma if exp.model.p_noise_dist is not None else np.zeros_like(exp.model.sysd.A)
        self.kf = KalmanFilter(exp.model.sysd.A, exp.model.sysd.B, exp.model.sysd.C, exp.model.sysd.D, self.kf_Q, exp.kf_R)
        self.kf_P = np.zeros_like(exp.model.sysd.A)

        self.x_update = None
        self.x_update_list = []
        self.y_list = []
        self.alarm_rate_list = []
        self.control_list = []
        self.alarm_list = []
        self.residual_list = []
        self.step_length = step_length
        self.alarm_rate_queue = queue.Queue(maxsize=step_length)
        self.y_queue = queue.Queue(maxsize=step_length)
        self.i = 0
        self.knn = joblib.load('save/knn.pkl')

    def __del__(self):
        self.exp.model.reset()
        print('__del__')

    # evolve
    def evolve(self):
        while not self.end_query:
            self.exp.model.update_current_ref(self.exp.ref[self.i])
            self.exp.model.evolve()
            if self.i == 0:
                self.x_update = self.exp.model.cur_x
            if self.i > 0:
                # self.exp.model.cur_y, end_query = self.exp.query.launch(self.exp.model.cur_y, self.exp.model.cur_index)
                self.x_update, P_update, residual = self.kf.one_step(self.x_update, self.kf_P, self.exp.model.cur_u, self.exp.model.cur_y)
                self.exp.model.cur_feedback = self.x_update
                self.kf_P = P_update
                # alarm = self.detector.detect(residual)
                # print(residual)
                alarm = self.detector.detect(residual)

                # # start recording
                # if self.exp.model.cur_index >= self.start_index - self.step_length:
                #     self.y_list.append(self.exp.model.cur_y)
                #     self.alarm_list.append(alarm)
                #
                #     if self.alarm_rate_queue.full():
                #         self.alarm_rate_queue.get()
                #         self.y_queue.get()
                #     self.alarm_rate_queue.put(alarm_rate)
                #     self.y_queue.put(self.exp.model.cur_y)
                #
                # # already achieve stable status
                # if self.exp.model.cur_index >= self.start_index:
                #     return 0

                # ready for attack
                if self.exp.model.cur_index >= self.start_index:
                    self.y_list.append(self.exp.model.cur_y)
                    return 0
            self.i += 1

    # launch attack once
    def evolve_once(self, delta_y):
        delta_y = delta_y.tolist()
        self.exp.model.cur_y = self.exp.model.cur_y + delta_y
        x_update, P_update, residual = self.kf.one_step(self.exp.model.cur_feedback, self.kf_P, self.exp.model.cur_u, self.exp.model.cur_y)
        self.exp.model.cur_feedback = x_update
        self.kf_P = P_update

        # print('residual')
        # print(residual)
        alarm = self.detector.detect(residual)
        alarm_rate = self.knn.predict_proba([delta_y])
        # if self.alarm_rate_queue.full():
        #     self.alarm_rate_queue.get()
        #     self.y_queue.get()
        # self.alarm_rate_queue.put(alarm_rate)
        self.alarm_list.append(alarm)
        self.alarm_rate_list.append(alarm_rate)

        self.exp.model.update_current_ref(self.exp.ref[self.i])
        self.exp.model.evolve()
        # self.y_queue.put(self.exp.model.cur_y)
        self.y_list.append(self.exp.model.cur_y)

        self.i += 1
        # return self.alarm_rate_queue, self.y_queue
        return self.alarm_rate_list, self.y_list


detector = chi_square(threshold=8.61)
detector = chi_square(threshold=21.86)
query = QueryOnce(detector=detector)
query.evolve()
print(query.y_list)

# delta_y = [-0.07503427, -0.14751052]
# query.evolve_once(delta_y)
# print(query.y_list)
# print(query.alarm_list)
# print(query.alarm_rate_list)
#
# delta_y = [0.4765643, 0.07827326]
# query.evolve_once(delta_y)
# print(query.y_list)
# print(query.alarm_list)
# print(query.alarm_rate_list)
#
# delta_y = [0.09, 0]
# for i in range(50):
#     query.evolve_once(delta_y)
# print(query.y_list)
# print(query.alarm_list)