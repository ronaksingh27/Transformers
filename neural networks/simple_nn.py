import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

class Model(nn.Module):

    def __init__(self,in_feature = 4, h1 = 5, h2 = 5, out_features = 3 ):
        #intantiates the nnModule class
        super().__init__()

        self.fc1 = nn.Linear(in_feature,h1)
        self.fc2 = nn.Linear(h1,h2)
        self.out = nn.Linear(h2,out_features)

    def forward( self , x ):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x
    
torch.manual_seed(41)

model = Model()#create an instance of the model

url = '/Users/ronaksingh/Desktop/Transformers/neural networks/iris.csv'
my_df = pd.read_csv(url)
print(my_df.head())

# my_df.head()

my_df['variety'] = my_df['variety'].replace({'Setosa':0,'Versicolor':1,'Virginica':2})
print(my_df.head())


############### TRAIN TEST SPLIT X,Y ############################

X = my_df.drop('variety',axis = 1)
y = my_df['variety']


X = X.values
y = y.values

print("y : ",y)
print("x : ",X)


#Train test split
X_train , X_test, y_train , y_test = train_test_split(X,y,test_size = 0.2,random_state = 41)


#Conver data to tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

#Set the criteria of model to measure the error , how far off the prediction is from the actual value
criterion = nn.CrossEntropyLoss()

#Choose Optimizer
optimizer = torch.optim.Adam(model.parameters(),lr = 0.01)


epochs = 100
losses = []
for i in range(epochs):

    #go forward and get a prediction
    y_pred = model.forward(X_train)

    #measure the loss/error , gonna be high at first
    loss = criterion(y_pred , y_train)

    losses.append(loss.detach().numpy())
    # print("epoch new : ",loss.detach().numpy())
    # print("epoch : ",loss)

    if i % 10 == 0 :
        print(f'Epoch {i} and loss is : {loss}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(range(epochs),losses)#x,y
plt.xlabel('Epochs')
plt.ylabel('Loss')


with torch.no_grad():
    y_val = model.forward(X_test)
    loss = criterion(y_val,y_test)

print(f'Loss : {loss}')


correct = 0
with torch.no_grad():
    for i,data in enumerate(X_test):
        y_val = model.forward(data)
            
        print(f'{i+1}.) {str(y_val)}  {y_val.argmax().item()} {y_test[i]}')

        if y_val.argmax().item() == y_test[i]:
            correct += 1

print(f'We got {correct} correct!')

new_iris = torch.tensor([5.6,3.7,2.2,0.5])
with torch.no_grad():
    print(model(new_iris))
    print(model(new_iris).argmax().item())

#SAVE THE NN MODEL
torch.save(model.state_dict(),'/Users/ronaksingh/Desktop/Transformers/neural networks/iris_model.pt')

#load the new model
new_model = Model()
new_model.load_state_dict(torch.load('/Users/ronaksingh/Desktop/Transformers/neural networks/iris_model.pt'))

print(new_model.eval())

new_iris = torch.tensor([5.6,3.7,2.2,0.5])
with torch.no_grad():
    print(new_model(new_iris))
    print(new_model(new_iris).argmax().item())