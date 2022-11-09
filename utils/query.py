import numpy as np


class Query:
    def __init__(self, y_up, y_low, K=4, start_index=200):
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

    def generate_square_points(self, index):
        for i in range(index, self.y_size):
            self.mask[i] = 0
            for j in range(self.K):
                temp = self.m * self.mask + self.y_low
                self.mask[i] += 1
                self.point.append(temp)
                self.generate_square_points(i + 1)

    def launch(self, cur_data, cur_index):
        self.point = np.array(self.point)
        end_query = False
        if self.index >= self.point.size:
            end_query = True
            return cur_data, end_query
        if cur_index >= self.start_index and self.index < self.point.size:

            cur_data = self.point[self.index]
            self.index += 1
            return cur_data, end_query
        else:
            return cur_data, end_query

    def query_point_reset(self):
        self.point = []

if __name__ == '__main__':
    A = np.array([1, 1])
    B = np.array([5, 5])
    query = Query(B, A)
