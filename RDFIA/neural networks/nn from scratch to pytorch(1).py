from tme5 import CirclesData
import torch

def init_params(nx, nh, ny):
    params = {}
    params["Wh"] = 0.3*torch.rand(nh,nx)
    params["Wy"] = 0.3*torch.rand(ny,nh)
    params["Bh"] = 0.3*torch.rand(nh)
    params["By"] = 0.3*torch.rand(ny)
    params["Wh"].requires_grad = True
    params["Wy"].requires_grad = True
    params["Bh"].requires_grad = True
    params["By"].requires_grad = True
    return params


def forward(params, X):
    outputs = {}
    outputs["X"] = X
    outputs["Htilde"] = torch.Tensor.matmul(X,params["Wh"].t()) + params["Bh"]
    outputs["H"] = torch.tanh(outputs["Htilde"])
    outputs["Ytilde"] = torch.Tensor.matmul(outputs["H"],params["Wy"].t()) + params["By"]
    outputs["Yhat"] = torch.softmax(outputs["Ytilde"],dim=1)
    return outputs['Yhat'], outputs

def loss_accuracy(Yhat, Y):
    L = -torch.mean(torch.sum(Y*torch.log(Yhat),dim=1),dim=0)
    _,indsY = torch.max(Y,dim=1)
    _,indsYhat = torch.max(Yhat,dim=1)
    acc = torch.sum(indsY == indsYhat,dtype=float)/len(Yhat)
    return L, acc

def sgd(params, eta):
    with torch.no_grad():
        params["Wy"] -= eta * params["Wy"].grad
        params["By"] -= eta * params["By"].grad
        params["Wh"] -= eta * params["Wh"].grad
        params["Bh"] -= eta * params["Bh"].grad
        params["Wy"].grad.zero_()
        params["By"].grad.zero_()
        params["Wh"].grad.zero_()
        params["Bh"].grad.zero_()
    return params

# data
data = CirclesData()
data.plot_data()
X_train = data.Xtrain
Y_train = data.Ytrain
X_test = data.Xtest
Y_test = data.Ytest

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
params = init_params(nx,nh,ny)
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
        Yhat , outputs = forward(params,Xb)
        # loss
        L,acc = loss_accuracy(Yhat,Yb)
        #backward
        L.backward()
        params = sgd(params,eta)
        
        # Compute train loss and accuracy 
        Yhat , outputs = forward(params,X_train)
        L,acc = loss_accuracy(Yhat,Y_train)
        loss_train.append(L)
        acc_train.append(acc)
        
        # Compute test loss and accuracy 
        Yhat , outputs = forward(params,X_test)
        L,acc = loss_accuracy(Yhat,Y_test)
        loss_test.append(L)
        acc_test.append(acc)
        
# Plot frontier for train and test
Ygrid,outputs = forward(params,data.Xgrid)
Ygrid[Ygrid>0.5] = 1
Ygrid[Ygrid<=0.5] = 0
Ygrid = Ygrid.detach()
data.plot_data_with_grid(Ygrid)

# Plot loss and accuracy for train and test
data.plot_loss(loss_train,loss_test,acc_train,acc_test)
    


