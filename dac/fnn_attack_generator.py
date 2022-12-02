import numpy as np
import torch
from query_once import QueryOnce
import joblib
from utils.detector.chi_square import chi_square
from sklearn.neighbors import KNeighborsClassifier


batch_size = 1


in_dim = 2
n_hidden_1 = 32
n_hidden_2 = 64
out_dim = 2

LR = 0.0001
EPOCH = 3

#
y_filter = [1, 0]
unsafe_set = [3.3, 3.5]
train_data = []
alarm_data = []
train_data_tensor = torch.tensor((), dtype=torch.float64)
alarm_data_tensor = torch.tensor((), dtype=torch.float64)


def reset():
    global train_data_tensor
    global alarm_data_tensor
    global alarm_data
    global train_data
    train_data = []
    alarm_data = []
    train_data_tensor = torch.tensor((), dtype=torch.float64)
    alarm_data_tensor = torch.tensor((), dtype=torch.float64)


def getInitialData():
    global train_data_tensor
    global query
    query.evolve()
    global train_data
    train_data.append(query.y_list[0])
    train_data_tensor = torch.tensor(train_data[0])


# get one step from the previews model state
def getOneStep(delta_y, step):
    global train_data_tensor
    global alarm_data_tensor
    global alarm_data
    global train_data
    global query
    query.evolve_once(delta_y)
    train_data.append(query.y_list[step + 1])
    train_data_tensor = torch.tensor(train_data[step + 1])
    alarm_data.append((query.alarm_rate_list[step][0][1]))
    alarm_data_tensor = torch.tensor(alarm_data[step])


def get_distance():
    y = train_data_tensor
    for i in y_filter:
        if i:
            temp = y[i]
    d1 = temp - unsafe_set[0]
    d2 = unsafe_set[1] - temp
    if d1 <= 0:
        return abs(d1)
    if d2 <= 0:
        return d2
    distance = abs(d1) if abs(d1) < abs(d2) else abs(d2)
    return distance


class custumLoss(torch.nn.Module):
    def __init__(self):
        super(custumLoss, self).__init__()

    def forward(self, delta_y, i):
        getOneStep(delta_y, i)
        # print('alarm_data_tensor')
        # print(alarm_data_tensor)
        custumLoss = alarm_data_tensor + get_distance()
        # print('get distance')
        # print(get_distance())
        return custumLoss


class simpleNet(torch.nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(simpleNet, self).__init__()
        self.layer1 = torch.nn.Linear(in_dim, n_hidden_1)
        self.layer2 = torch.nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = torch.nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


model = simpleNet(in_dim, n_hidden_1, n_hidden_2, out_dim)
if torch.cuda.is_available():
    model = model.cuda()

# loss and optimizer
loss_func = custumLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# train
for step in range(EPOCH):
    reset()
    detector = chi_square(threshold=8.61)
    query = QueryOnce(detector=detector)
    getInitialData()
    print('initial training data')
    print(train_data_tensor)

    print('step:' + str(step))
    for i in range(100):
        if torch.cuda.is_available():
            input_data = train_data_tensor.cuda().float()
        else:
            input_data = train_data_tensor.float()

        # output = model(torch.unsqueeze(input_data, dim=2))
        output = model(input_data)
        loss = loss_func(torch.squeeze(output), i)
        # print('loss')
        # print(loss)
        optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        print(i, loss.cpu())
        if i % 49:
            torch.save(model, 'save/model.pkl')

torch.save(model, 'save/model.pkl')