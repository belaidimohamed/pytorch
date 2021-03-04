import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

import network

net = network.Net()
training_data = np.load("training_data.npy",allow_pickle=True)


# ------------------------------------------------- Config ----------------------------------------------------

optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in training_data]).view(-1,50,50)
X = X/255.0
Y = torch.Tensor([i[1] for i in training_data])
VAL_PCT = 0.1  # lets reserve 10% of our data for validation
val_size = int(len(X)*VAL_PCT)

train_X = X[:-val_size]
train_y = Y[:-val_size]

test_X = X[-val_size:]
test_y = Y[-val_size:]

print(len(training_data))
print(len(train_X), len(test_X))

# ------------------------------------------------- Training ----------------------------------------------------


BATCH_SIZE = 100
EPOCHS = 2

for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)): # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
        #print(f"{i}:{i+BATCH_SIZE}")
        batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
        batch_y = train_y[i:i+BATCH_SIZE]
        net.zero_grad()

        outputs = net(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()    # Does the update
    print(f"Epoch: {epoch}. Loss: {loss}")


# ------------------------------------------------- Tests ----------------------------------------------------


correct = 0
total = 0
with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        real_class = torch.argmax(test_y[i])
        net_out = net(test_X[i].view(-1, 1, 50, 50))[0]  # returns a list,
        predicted_class = torch.argmax(net_out)

        if predicted_class == real_class:
            correct += 1
        total += 1
print("Accuracy : ", round(correct/total, 3))











