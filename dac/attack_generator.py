import numpy as np
import torch
from query_once import QueryOnce
import joblib
from utils.detector.chi_square import chi_square
from sklearn.neighbors import KNeighborsClassifier

input_size = 2
output_size = 2
hidden_size = 64
batch_size = 1
num_layers = 1

LR = 0.0001
EPOCH = 10

#
y_filter = [1, 0]
unsafe_set = [3.3, 3.5]
train_data = []
alarm_data = []
train_data_tensor = torch.tensor((), dtype=torch.float64)
alarm_data_tensor = torch.tensor((), dtype=torch.float64)

def getInitialData():
    global train_data_tensor
    query.evolve()
    train_data.append(query.y_list[0])
    train_data_tensor = torch.tensor(train_data)

# get one step from the previews model state
def getOneStep(delta_y, step):
    global train_data_tensor
    global alarm_data_tensor
    query.evolve_once(delta_y)
    train_data.append(query.y_list[step + 1])
    train_data_tensor = torch.tensor(train_data[step + 1])
    alarm_data.append((query.alarm_rate_list[step]))
    alarm_data_tensor = torch.tensor(alarm_data[step])

def get_distance():
    y = train_data_tensor
    for i in y_filter:
        if i:
            temp = y[i]
    d1 = temp - unsafe_set[0]
    d2 = unsafe_set[1] - temp
    if d1 <= 0:
        return d1
    if d2 <= 0:
        return d2
    distance = abs(d1) if abs(d1) < abs(d2) else abs(d2)
    return distance

class custumLoss(torch.nn.Module):
    def __init__(self):
        super(custumLoss, self).__init__()

    def forward(self, delta_y, i):
        getOneStep(delta_y, i)
        custumLoss = alarm_data_tensor + get_distance()
        return custumLoss

class simpleLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(simpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.out = torch.nn.Sequential(torch.nn.Linear(hidden_size, output_size))

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)

        # forward propagate lstm
        out, (h_n, h_c) = self.lstm(x, None)
        print('out')
        print(out)

        # select the last step as output
        out = self.out(out[:, -1, :])
        return out


detector = chi_square(threshold=8.61)
query = QueryOnce(detector=detector)
# train_data_tensor = getInitialData()
getInitialData()
print('initial training data')
print(train_data_tensor)

# delta_y = [0.01503427, 0.01751052]
# # train_data_tensor, alarm_data_tensor = getOneStep(delta_y, 0, train_data_tensor)
# train_data_tensor, alarm_data_tensor = getOneStep(delta_y, 0)
# print(train_data_tensor)
# print(alarm_data_tensor)

model = simpleLSTM(input_size, hidden_size, num_layers, output_size)
if torch.cuda.is_available():
    model = model.cuda()

# loss and optimizer
loss_func = custumLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# train
for step in range(EPOCH):
    for i in range(100):
        if torch.cuda.is_available():
            input_data = train_data_tensor[0].cuda()
        else:
            input_data = train_data_tensor[0]

        output = model(torch.unsqueeze(input_data, dim=2))
        loss = loss_func(torch.squeeze(output), i)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(step, loss.cpu())
        if step % 10:
            torch.save(model, 'model.pkl')

torch.save(model, 'model.pkl')