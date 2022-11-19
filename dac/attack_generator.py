import torch

input_size = 2
output_size = 2
hidden_size = 64
batch_size = 1
num_layers = 1

LR = 0.0001
EPOCH = 10

y_filter = []
unsafe_set = [-20, 20]

train_loader

## get the first 10 steps from steady state
def getData():

## get one step from the previews model state
def getOneStep():

def get_distance(y):


class custumLoss(torch.nn.Module):
    def __init__(self):
        super(custumLoss, self).__init__()

    def forward(self, alarm_rate, y):
        custumLoss = alarm_rate + get_distance(y)
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

        # select the last step as output
        out = self.fc(out[:, -1, :])
        return out


model = simpleLSTM(input_size, hidden_size, num_layers, output_size)
if torch.cuda.is_available():
    model = model.cuda()

# loss and optimizer
loss_func = torch.nn.custumLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for step in range(EPOCH):
    for input in train_loader:
        if torch.cuda.is_available():
            input = input.cuda()

        output = model(torch.unsqueeze(input, dim=2))
        loss = loss_func(torch.squeeze(output))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(step, loss.cpu())
        if step % 10:
            torch.save(model, 'model.pkl')
torch.save(model,'model.pkl')