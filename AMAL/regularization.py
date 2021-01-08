
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader 
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
from collections import OrderedDict
  
#    DATASET, MODEL, FUNCTIONS FOR TRAINING AND FOR GLOBAL LOSS
   
class Dataset(Dataset):
    def __init__(self,X,Y,mean,std):
        self.Y = Y
        #normalizing
        self.X = X.float()
        self.X = self.X.view(len(X),-1)
        self.X -= mean
        self.X /= std
    def __getitem__(self,index):
        return self.X[index],self.Y[index]
    def __len__(self):
        return len(self.X)

def store_approx_grad(var):
    def hook(grad):
        var.grad = grad
    return var.register_hook(hook) 

def train_one_epoch(data,model,loss_fn,device,norm=0,lambda_reg=0):
    ''' Precise norm L1 or L2 and the hyperparameter lambda '''
    for x,y in train_loader:
        #zero grad
        optimizer.zero_grad()
        #gpu and forward
        x = x.to(device)
        y = y.to(device)
        yhat_proba = model.forward(x) 
        yhat = torch.log(yhat_proba)
        #loss
        loss = loss_fn(yhat,y)
        #regularization
        if norm:
            reg = torch.zeros((1,))
            #accumulate the norm of each layer
            for name, param in model.named_parameters():
                if 'weight' in name:
                    reg += torch.norm(param,norm).item()
            reg = reg.to(device)
            loss = loss + lambda_reg*reg
        #backpropagation
        loss.backward()
        #update
        optimizer.step()
        
def global_loss(data,model,loss_fn,device,writer,histo=0):
    ''' Precise histo to save the histogram for entropy and layers weights '''
    loss_data = torch.zeros((1,)).to(device) #accumulate the loss across the batches
    N_iter = 0
    for x,y in data:
        N_iter += 1
        #GPU and forward
        x = x.to(device)
        y = y.to(device)
        yhat_proba = model.forward(x) # cross-entropy in two steps to compute entropy
        yhat = torch.log(yhat_proba)
        #loss
        loss = loss_fn(yhat,y)
        loss_data += loss
    if histo:
        train_mode = model.training
        #if training
        if train_mode:
            #add entropy (only from a batch)
            cross_entropy = torch.distributions.categorical.Categorical(probs = yhat_proba).entropy()  
            writer.add_histogram('Entropy train', cross_entropy, histo)
            #add layers weights
            j = 0
            for name, param in model.named_parameters():
                if 'weight' in name:
                    j+=1
                    writer.add_histogram(name,param.data, histo)
        #if testing
        else:
            #add entropy (only from a batch)
            cross_entropy = torch.distributions.categorical.Categorical(probs = yhat_proba).entropy()  
            writer.add_histogram('Entropy test', cross_entropy, histo)
    return loss_data.item()/N_iter
   
p = 0.05
#data
dataset = MNIST('./data',train=True,download=True)
train_images , train_labels = dataset.train_data,dataset.train_labels
test_images, test_labels = dataset.test_data,dataset.test_labels
X = train_images[:int(p*len(train_images))]
#normalizing with mean and std of train images
mean = X.float().view(len(X),-1).mean()
std = X.float().view(len(X),-1).std()
train_data = Dataset(X,train_labels[:int(p*len(train_images))],mean,std)
test_data = Dataset(test_images[:int(p*len(train_images))],test_labels[:int(p*len(train_images))],mean,std)
   

class ClassMNIST(nn.Module):
    def __init__(self,D_in,D_hidden,D_out):
        super(ClassMNIST,self).__init__()
        self.layers = nn.Sequential(OrderedDict([
            #('layernorm', nn.LayerNorm(D_in)),
            ('linear1', nn.Linear(D_in,D_hidden)),
            #('batchnorm1', nn.BatchNorm1d(100)),
            ('relu1', nn.ReLU()),
            ('linear2', nn.Linear(D_hidden,D_hidden)),
            #('batchnorm2', nn.BatchNorm1d(100)),
            ('relu2', nn.ReLU()),
            ('dropout', nn.Dropout(p=0.5)),
            ('linear3', nn.Linear(D_hidden,D_out))
        ]))
        
    def forward(self,x):
        out = self.layers(x)
        return nn.functional.softmax(out,dim=1)
 
#params dataloader
nb_epoch = 500
batch_size = 300
train_loader = DataLoader(train_data,shuffle=True,batch_size=batch_size)
test_loader = DataLoader(test_data,shuffle=True,batch_size=batch_size)

#params model
D_in,D_hidden,D_out = len(train_images[0])*len(train_images[0][0]),100,10
lr = 1e-4
weight_decay = 1e-5

#model
model = ClassMNIST(D_in,D_hidden,D_out)
optimizer = torch.optim.Adam(model.parameters(),lr=lr)
loss_fn = torch.nn.NLLLoss()

#gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
loss_fn = loss_fn.to(device)
   
name = 'runs/dropout'
writer = SummaryWriter(log_dir=name)
#gradient descent
for i in range(1,nb_epoch+1):
    #training
    train_one_epoch(train_loader,model,loss_fn,device,norm=0,lambda_reg=0)
    if i%100 == 0:
        print("Iter {}".format(i))
        histo = i//100
    else:
        histo = 0    
    #global loss for train and test at each epoch, every 100 epochs for histogram of weights and entropy
    writer.add_scalar('Train_cost', global_loss(train_loader,model,loss_fn,device,writer,histo), i)
    model.eval()
    writer.add_scalar('Test_cost', global_loss(test_loader,model,loss_fn,device,writer,histo), i)
    model.train()
writer.close()
   
   