from tme5 import CirclesData
import torch
import torch.nn as nn

def init_model(nx,nh,ny):
    model = nn.Sequential(
                nn.Linear(nx,nh),
                nn.Tanh(),
                nn.Linear(nh,ny),
            )
    loss = nn.CrossEntropyLoss()
    return model,loss

def sgd(model, eta):
    with torch.no_grad():
        for param in model.parameters():
            param -= eta * param.grad
    model.zero_grad()
    
def accuracy(Yhat,Y):
    # Y of shape (N) not one-shot
    _,indsYhat = torch.max(Yhat,dim=1)
    acc = torch.sum(indsYhat == Y,dtype=float)/len(Yhat)
    return acc

### data
data = CirclesData()
data.plot_data()
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
Nbatch = 10
Nepoch = 200
eta = 0.1

# init
model,loss = init_model(nx,nh,ny)
loss_train = []
acc_train = []
loss_test = []
acc_test = []

for i in range(Nepoch):
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
        L.backward()
        sgd(model,eta)
        
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

# Plot frontier for train and test
Ygrid = model(data.Xgrid)
Ygrid[Ygrid>0.5] = 1
Ygrid[Ygrid<=0.5] = 0
Ygrid = Ygrid.detach()
data.plot_data_with_grid(Ygrid)

# Plot loss and accuracy for train and test
data.plot_loss(loss_train,loss_test,acc_train,acc_test)
    
    
    

