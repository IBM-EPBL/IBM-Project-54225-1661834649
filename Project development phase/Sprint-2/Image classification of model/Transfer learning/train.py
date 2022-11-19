from preprocessing import *
import time

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import models

lr = 0.001
device = torch.device("cuda")

train_loader = trainDataset()
test_loader = testDataset()

model = models.resnet50().cuda()
########################
# model.load_state_dict(torch.load('checkpoint/checpoint_epoch.pt'))
# model.eval()
########################
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

def train(epoch):
    model.train()
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if idx % 10 == 0:
            print("epoch: ", epoch, "  process: ", int((idx / len(train_loader)) * 100),
                "%  Loss: ", loss.data.item())
    
    torch.save(model.state_dict(), 'checkpoint/epoch'+str(epoch)+'.pt')

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).data.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
    
    test_loss /= len(test_loader.dataset)
    print("Average Loss: ", test_loss, "  Accuracy: ", correct, " / ",
    len(test_loader.dataset), "  ", int(correct / len(test_loader.dataset) * 100), "%")


if __name__ == "__main__":
    for epoch in range(5):
        start = time.time()
        train(epoch)
        end = time.time()
        print("It takes ", end - start, " seconds")
        test()
