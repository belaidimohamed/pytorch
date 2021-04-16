import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import network
import os
import time
import sys

img_resolution = 50

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

net = network.Net().to(device)
training_data = np.load(r"C:\Users\mohamed\Desktop\tensor\training_data.npy",allow_pickle=True)

# ------------------------------------------------- Config ----------------------------------------------------

X = torch.Tensor([i[0] for i in training_data]).view(-1,img_resolution,img_resolution)
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


optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

def smoothingLabel(outputs, target ,smoothing_coefficient) :
    weight = outputs.new_ones(outputs.size()) * smoothing_coefficient / (outputs.size(-1) - 1.)
    target = torch.from_numpy(np.where(target.cpu().numpy() == 1)[1]).to(device)
    weight.scatter_(-1, target.unsqueeze(-1), (1. - smoothing_coefficient))
    losses= -weight * outputs
    return losses.sum(dim=-1).mean()

# ------------------------------------------------- Training ----------------------------------------------------

def train(net,BATCH_SIZE=256,EPOCHS=16):
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        for i in range(0, len(train_X), BATCH_SIZE):

            batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, img_resolution, img_resolution)
            batch_y = train_y[i:i+BATCH_SIZE]

            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            net.zero_grad()

            optimizer.zero_grad()   # zero the gradient buffers
            outputs = net(batch_X)

            loss = smoothingLabel(outputs, batch_y,0.12)
            # loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()    # Does the update

        print(f"Epoch: {epoch}. Loss: {loss}")
train(net)

# ------------------------------------------------- Tests ----------------------------------------------------

test_X.to(device)
test_y.to(device)

def test(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_y[i]).to(device)
            net_out = net(test_X[i].view(-1, 1, img_resolution, img_resolution).to(device))[0]  # returns a list,
            predicted_class = torch.argmax(net_out)

            if predicted_class == real_class:
                correct += 1
            total += 1

    print("Accuracy: ", round(correct/total, 3))
    response = input('save me or no (y/n) ? : ')
    while 1:
        if response == 'n':
            break
        elif response == 'y':
            torch.save(net.state_dict(), os.path.join(r"C:\Users\mohamed\Desktop\tensor\trained_models",str(round(correct/total, 3))+"#dogVscat" ))
            print('saving done !')
            break
        else :
            pass



test(net)








#MSELoss 0.764
#sm = 0.1  0.757







