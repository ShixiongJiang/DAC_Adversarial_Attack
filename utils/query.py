import numpy as np
from collections import deque
from scipy.spatial import distance
import random
from scipy.spatial import distance


class Query:
    def __init__(self, y_up, y_low, K=4, start_index=200, N_step=1):
        self.point = []
        self.y_up = y_up
        self.y_low = y_low
        self.y_size = self.y_up.size
        self.K = K
        self.m = (y_up - y_low) / self.K
        self.mask = np.zeros(self.y_size)
        self.start_index = start_index
        self.index = 0
        self.generate_square_points(0)
        self.model = None
        self.NEG = False
        self.POS = True
        self.e = 0.1
        self.pts_near_b = []
        self.pts_near_b_labels = []
        self.collect_query = None
        self.N_step = N_step
        self.end_query=False

    def set_model(self, model):
        self.model = model

    def generate_square_points(self, index):
        for i in range(index, self.y_size):
            self.mask[i] = 0
            for j in range(self.K):
                temp = self.m * self.mask + self.y_low
                self.mask[i] += 1
                self.point.append(temp)
                self.generate_square_points(i + 1)

    def launch(self, cur_data, cur_index, query_type='square'):
        if query_type == 'active_learn':
            if self.collect_query is None:
                self.collect_query = np.array(self.pts_near_b)
            if self.index >= len(self.collect_query):
                self.end_query = True
                self.index = 0
                self.collect_query = None
                return cur_data, self.end_query
            if cur_index == self.start_index and self.index < len(self.collect_query):
                a = (cur_index - self.start_index ) * self.y_size
                b = a + self.y_size
                # cur_data = self.collect_query[self.index][a:b]
                cur_data = self.collect_query[self.index][a:b] + cur_data
                self.index += 1
                return cur_data, self.end_query
            elif cur_index > self.start_index:
                a = (cur_index - self.start_index) * self.y_size
                b = a + self.y_size
                # cur_data = self.collect_query[self.index][a:b]
                cur_data = self.collect_query[self.index][a:b] + cur_data
                return cur_data, self.end_query
            else:
                return cur_data, self.end_query
        else:
            if self.collect_query is None:
                self.collect_query = np.array(self.point)
            if self.index >= len(self.collect_query):
                self.end_query = True
                self.index = 0
                self.collect_query = None
                return cur_data, self.end_query
            if cur_index == self.start_index and self.index < len(self.collect_query):
                # cur_data = self.query_near(cur_data)
                cur_data = self.collect_query[self.index]
                self.index += 1
                return cur_data, self.end_query
            elif cur_index > self.start_index:
                cur_data = self.query_near(cur_data)
                return cur_data, self.end_query
            else:
                return cur_data, self.end_query


        # if self.collect_query is None:
        #     if query_type == 'active_learn':
        #         self.collect_query = np.array(self.pts_near_b)
        #     else:
        #         self.collect_query = np.array(self.point)
        #
        # if self.index >= len(self.collect_query):
        #     self.end_query = True
        #     self.index = 0
        #     self.collect_query = None
        #     return cur_data, self.end_query
        # if cur_index == self.start_index and self.index < len(self.collect_query):
        #     # if self.N_step == 1:
        #     if query_type == 'active_learn':
        #         cur_data = self.collect_query[self.index]
        #     else:
        #         cur_data = self.query_near(cur_data)
        #     self.index += 1
        #     return cur_data, self.end_query
        # elif cur_index > self.start_index:
        #     cur_data = self.query_near(cur_data)
        #     return cur_data, self.end_query
        # else:
        #     return cur_data, self.end_query

    def query_point_reset(self):
        self.point = []

    def random_vector(self, label=None):

        def rv_gen():
            temp = None
            for i in range(self.N_step):
                if temp is None:
                    temp = np.random.rand(self.y_size) - 0.5
                    temp = temp * (self.y_up - self.y_low)
                    # temp = temp * (self.y_up - self.y_low) + self.y_low
                else:
                    a = np.random.rand(self.y_size) - 0.5
                    a = a * (self.y_up - self.y_low)
                    # a = a * (self.y_up - self.y_low) + self.y_low
                    temp = np.concatenate((temp, a))
            return temp

        if label is not None:
            while True:
                a = rv_gen()
                l = self.local_query(a)
                if l == label:
                    return a
        else:
            return rv_gen()

    def active_learn(self, m):  # m is the number of query to learn
        pts_near_b_in_x = []
        pts_near_b_in_x_label = []
        for i in range(m):
            x1 = self.random_vector(label=0)
            x2 = self.random_vector(label=1)
            xb1, xb2 = self.push_to_b(x1, x2, self.e)
            pts_near_b_in_x.append(xb1)
            pts_near_b_in_x.append(xb2)
            pts_near_b_in_x_label.extend((self.NEG, self.POS))
        self.pts_near_b = pts_near_b_in_x
        self.pts_near_b_labels = pts_near_b_in_x_label
        return pts_near_b_in_x, pts_near_b_in_x_label

    def push_to_b(self, xn, xp, e):
        assert self.local_query(xn) == self.NEG
        assert self.local_query(xp) == self.POS

        d = distance.euclidean(xn, xp) / \
            distance.euclidean(np.ones(self.y_size), np.zeros(self.y_size))
        if d < e:
            print(f'bin search done with %f', d)
            return xn, xp
        mid = .5 * np.add(xn, xp)

        l = self.local_query(mid)
        if l == self.NEG:
            return self.push_to_b(mid, xp, e)
        else:
            return self.push_to_b(xn, mid, e)

    def local_query(self, x):
        if self.model is None:
            raise Exception("model is None, set model")
        return self.model.predict(x.reshape(1, -1))

    def query_near(self, cur_y):


        a = (np.random.rand(*cur_y.shape) - 0.5) / 4 * cur_y + cur_y
        # a = rand * cur_y

        return a


if __name__ == '__main__':
    A = np.array([1, 1])
    B = np.array([5, 5])
    query = Query(B, A)
    x, label = query.active_learn(10)
    print(x)
