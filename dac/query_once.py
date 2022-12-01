import numpy as np
from settings_baseline import motor_speed_bias, quadruple_tank_bias, lane_keeping, f16_bias, aircraft_pitch_bias, boeing747_bias, platoon_bias, quadrotor_bias, rlc_circuit_bias
from utils.observers.kalman_filter import KalmanFilter
from utils.detector.cusum import CUSUM
import queue
import joblib

# input attack_y
# output query result, y, ref
class QueryOnce:
    def __init__(self, start_index=100, exp=quadruple_tank_bias, detector=CUSUM, step_length=10):
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
        self.control_list = []
        self.alarm_list = []
        self.residual_list = []
        self.step_length = step_length
        self.alarm_rate_queue = queue.Queue(maxsize=step_length)
        self.y_queue = queue.Queue(maxsize=step_length)
        self.i = 0

    # evolve
    def evolve(self):
        while not self.end_query:
            self.exp.model.update_current_ref(self.exp.ref[self.i])
            self.exp.model.evolve()
            if self.i == 0:
                self.x_update = self.exp.model.cur_x
            if self.i > 0:
                # self.exp.model.cur_y, end_query = self.exp.query.launch(self.exp.model.cur_y, self.exp.model.cur_index)
                x_update, P_update, residual = self.kf.one_step(x_update, self.kf_P, self.exp.model.cur_u, self.exp.model.cur_y)
                self.exp.model.cur_feedback = x_update
                self.kf_P = P_update
                # alarm = self.detector.detect(residual)
                alarm_rate, alarm = self.detector.detect(residual)

                # start recording
                if self.exp.model.cur_index >= self.start_index - self.step_length:
                    self.y_list.append(self.exp.model.cur_y)
                    self.alarm_list.append(alarm)

                    if self.alarm_rate_queue.full():
                        self.alarm_rate_queue.get()
                        self.y_queue.get()
                    self.alarm_rate_queue.put(alarm_rate)
                    self.y_queue.put(self.exp.model.cur_y)

                # already achieve stable status
                if self.exp.model.cur_index >= self.start_index:
                    return 0
            self.i += 1

    # launch attack once
    def evolve_once(self, cur_data, cur_index, delta_y):
        self.exp.model.cur_y = self.exp.model.cur_y + delta_y
        x_update, P_update, residual = self.kf.one_step(self.exp.model.cur_feedback, self.kf_P, self.exp.model.cur_u, self.exp.model.cur_y)
        self.exp.model.cur_feedback = x_update
        self.kf_P = P_update

        alarm_rate, alarm = self.detector.detect(residual)
        if self.alarm_rate_queue.full():
            self.alarm_rate_queue.get()
            self.y_queue.get()
        self.alarm_rate_queue.put(alarm_rate)

        self.exp.model.update_current_ref(self.exp.ref[self.i])
        self.exp.model.evolve()
        self.y_queue.put(self.exp.model.cur_y)

        self.i += 1
        return self.alarm_rate_queue, self.y_queue

query = QueryOnce()
query.evolve()
print('y_queue' + query.y_queue)
print('alarm_rate_queue' + query.alarm_rate_queue)