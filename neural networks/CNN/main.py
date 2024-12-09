import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import ssl
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# %matplotlib inline


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


#Convert MNIST Image files into a to tensor of 4 dimensions( #of images , Height, Width, Channel)
transform = transforms.ToTensor()

#Train data
train_data = datasets.MNIST(root = "/Users/ronaksingh/Desktop/Transformers/neural networks/CNN/cnn_data" , train = True , download = True , transform = transform)
print("train data :" , train_data)

#Test data
test_data = datasets.MNIST(root = "/Users/ronaksingh/Desktop/Transformers/neural networks/CNN/cnn_data" , train = False , download = True , transform = transform)
print("test data : " , test_data)



############################################## CONVOLUTIONAL-AND-POOLING-LAYERS ##############################################

#Create a small batch size for images lets say 10
train_loader = DataLoader(train_data , batch_size = 10 , shuffle = True)    
test_loader = DataLoader(test_data , batch_size = 10 , shuffle = False)

#example of Convolutional layer( 2 Layers)
conv1 = nn.Conv2d(1,6,3,1)#in_images,  out_layers/using_6diff_filters, kernel_size, stride
conv2 = nn.Conv2d(6,16,3,1)


#Grab the 1st MNIST image
for i , (X_train , y_train) in enumerate(train_data):
    break

print("X_train shape : " , X_train.shape)

x = X_train.view(1,1,28,28)

x = F.relu(conv1(x))
print("After 1st Convolution : " , x.shape)


#Pass through pooling layer
x = F.max_pool2d(x,2,2)#kernel of 2 and stride 2
print("After 1st Pooling : " , x.shape)#26/2(stride len) = 13

#Pass through 2nd Convolutional layer
x = F.relu(conv2(x))
print("After 2nd Convolution : " , x.shape)

#Pass through 2nd Pooling layer
x = F.max_pool2d(x,2,2)
print("After 2nd Pooling : " , x.shape)#11/2(stride len) = 5.5 = 5( round down image)


############################################## CNN-Model ##############################################

class ConvolutionalNetwork(nn.Module):
    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(1,6,3,1)
        self.conv2 = nn.Conv2d(6,16,3,1)

        #Fully Connected Layer
        self.fc1 = nn.Linear(5*5*16,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X,2,2)

        #Second Pass
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X,2,2)

        #Flatten the image
        X = X.view(-1,5*5*16)#-1 so that we can vary batch size

        #Fully connected layers
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return F.log_softmax(X,dim = 1)



#Create an instance of the model
torch.manual_seed(41)

model = ConvolutionalNetwork()
print(model)

#Loss function Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.01)


start_time = time.time()

#Create variable to track things 
epochs = 5
train_losses = []
test_losses = []
train_correct = []
test_correct = []

#For loop for epochs
for i in range(epochs):
    trn_correct = 0
    tst_correct = 0


    #Train
    for b,(X_train,y_train) in enumerate(train_loader):
        b+=1

        y_pred = model(X_train)
        loss = criterion(y_pred,y_train)

        predicted = torch.max(y_pred.data,1)[1]#gets the predicted value

        batch_corr = (predicted == y_train).sum()
        trn_correct += batch_corr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b%600 == 0:
            print("predicted : ",predicted)
            print("y_train : ",y_train)
            print(f"Epoch : {i} Batch : {b} Loss : {loss.item()}")

    train_losses.append(loss)
    train_correct.append(trn_correct)


with torch.no_grad():
    for b,(X_test,y_test) in enumerate(test_loader):
        y_val = model(X_test)
        predicted = torch.max(y_val.data,1)[1]
        tst_correct += (predicted == y_test).sum()

loss = criterion(y_val,y_test)
test_losses.append(loss)
test_correct.append(tst_correct)






total = time.time() - start_time
print(f"Training took : {total/60} minutes")


