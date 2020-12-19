from tme5 import MNISTData
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def init_model(nx,nh,ny,eta):
    model = nn.Sequential(
                nn.Linear(nx,nh),
                nn.Tanh(),
                nn.Linear(nh,ny),
            )
    loss = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(),lr=eta)
    return model,loss,optim

def accuracy(Yhat,Y):
    # Y of shape (N) not one-shot
    _,indsYhat = torch.max(Yhat,dim=1)
    acc = torch.sum(indsYhat == Y,dtype=float)/len(Yhat)
    return acc

### data
data = MNISTData()
## data train
X_train = data.Xtrain
# transform Y_train to argmax and type long
Y_train = data.Ytrain
_,Y_train = torch.max(Y_train,dim=1)
Y_train = Y_train.long()

### data test
X_test = data.Xtest
## transform Y_test to argmax and type long
Y_test = data.Ytest
_,Y_test = torch.max(Y_test,dim=1)
Y_test = Y_test.long()

# dimensions
N = data.Xtrain.shape[0]
nx = data.Xtrain.shape[1]
ny = data.Ytrain.shape[1]
nh = 5

# parameters
Nbatch = 20
Nepoch = 10
eta = 0.001

# init
model,loss,optim = init_model(nx,nh,ny,eta)
loss_train = []
acc_train = []
loss_test = []
acc_test = []

for i in range(Nepoch):
    print("epoch {}".format(i+1))
    indices = torch.randperm(N)
    for j in range(N//Nbatch):
        
        # batch
        ind = indices[j*Nbatch:(j+1)*Nbatch]
        Xb,Yb = X_train[ind],Y_train[ind]
        # forward
        Yhat = model(Xb)
        # loss
        L = loss(Yhat,Yb)
        #backward
        optim.zero_grad()
        L.backward()
        optim.step()
        
        if j%500 == 0:
            print("batch {}".format(j))
            # Compute train loss and accuracy 
            Yhat = model(X_train)
            L = loss(Yhat,Y_train)
            loss_train.append(L)
            acc_train.append(accuracy(Yhat,Y_train))
            
            # Compute test loss and accuracy 
            Yhat = model(X_test)
            L = loss(Yhat,Y_test)
            loss_test.append(L)
            acc_test.append(accuracy(Yhat,Y_test))

plt.figure()
plt.plot(np.array(acc_train), label="acc. train")
plt.plot(np.array(acc_test), label="acc. test")

plt.figure()
plt.plot(np.array(loss_train), label="loss train")
plt.plot(np.array(loss_test), label="loss test")
