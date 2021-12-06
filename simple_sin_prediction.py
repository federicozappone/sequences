import torch
import torch.nn as nn
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler


# convert passengers data to float
all_data = []

for x in range(144):
    all_data.append(math.sin(x))

all_data = np.array(all_data)

print("sin data (float)")
print(all_data)

# use the last 12 months for testing
test_data_size = 12

train_data = all_data[:-test_data_size]
test_data = all_data[-test_data_size:]

print("test data")
print(test_data)

print("train data len", len(train_data))
print("test data len", len(test_data))

# normalize train data
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))

# convert to tensor
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

# set training window (months)
train_window = 12


def create_inout_sequences(input_data, train_window):
    inout_seq = []
    input_data_len = len(input_data)
    for i in range(input_data_len - train_window):
        train_seq = input_data[i:i + train_window]
        train_label = input_data[i + train_window:i + train_window + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


# create sequences
train_inout_seq = create_inout_sequences(train_data_normalized, train_window)


# define model
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


# create model, loss and optimizer
model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("model")
print(model)


epochs = 100

for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i%10 == 0:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')


# evaluate

fut_pred = 12

# get the last 12 values from the training set to predict the next 12
test_inputs = train_data_normalized[-train_window:].tolist()

print("test inputs")
print(test_inputs)


model.eval()

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        test_inputs.append(model(seq).item())

# print future prediction

print("future prediction")
print(test_inputs[fut_pred:])

# rescale output data

actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:]).reshape(-1, 1))

print("actual predictions")
print(actual_predictions)

# plot flight data

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 5
plt.rcParams["figure.figsize"] = fig_size

# plot predicted data against original data

x = np.arange(len(all_data) - 12, len(all_data), 1)


plt.title("Sin Function")
plt.ylabel("y")
plt.grid(True)
plt.autoscale(axis="y", tight=True)
plt.plot(all_data)
plt.plot(x, actual_predictions)

plt.show()


plt.title("Sin Function")
plt.ylabel("y")
plt.grid(True)
plt.autoscale(axis="y", tight=True)

plt.plot(all_data[-train_window:])
plt.plot(x, actual_predictions)

plt.show()
