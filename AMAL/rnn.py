# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 11:22:33 2019

@author: Jp
"""

import csv
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

#get data
data = csv.reader("tempAMAL_train.csv")
data = pd.read_csv("tempAMAL_train.csv",engine='python')
data = data.drop('datetime',axis=1)
data = data.values

#process missing data by replacing with the previous temperature
for i in range(len(data)):
    for j in range(len(data[0])):
        if np.isnan(data[i,j]):
            data[i,j] = data[i-1,j]
mean,std = data.mean(),data.std()
data = (data-mean)/std

#RNN
class RNN(nn.Module):
    def __init__(self,D_in,D_hidden,D_out,hidden_activ):
        super(RNN,self).__init__()
        self.linear_hidden = nn.Linear(D_in+D_hidden,D_hidden)
        self.linear_out = nn.Linear(D_hidden,D_out)
        self.hidden_activ = hidden_activ
        self.list_h = []
        
    def one_step(self,x):
        return self.hidden_activ(self.linear_hidden(torch.cat((x,self.h),dim=1)))
            
    def forward(self,x):
        self.list_h = []
        self.h = torch.zeros((x.shape[1],D_hidden))
        for i in range(len(x)):
            self.h = self.one_step(x[i])
            self.list_h.append(self.h)
                        
    def decoder(self,horizon):
        yhat = self.linear_out(self.list_h[-horizon])
        for i in range(1,horizon):
            yhat = torch.cat((yhat,self.linear_out(self.list_h[-horizon+i])),dim=1)
        return yhat
                
    
##data train and test
length = 100
batch_size = 10
N = 5000
nb_cities = 5
#ind_cities = np.random.choice(list(range(len(data[0]))),size = nb_cities,replace = False)
ind_cities = [0,1,2,3,4]
ind_time = torch.randint(len(data)-length,size=(1,))
ind_city = torch.randint(len(ind_cities),size=(1,))
datax_train = torch.Tensor(data[ind_time:ind_time+length,ind_city])
datay_train = [ind_city]
for i in range(N-1):
    ind_time = torch.randint(len(data)-length,size=(1,))
    ind_city = torch.randint(len(ind_cities),size=(1,))
    datay_train.append(ind_city)
    datax_train = torch.cat((datax_train,torch.Tensor(data[ind_time:ind_time+length,ind_city])))
datay_train = torch.LongTensor(datay_train)
datax_train = datax_train.view(length,-1,1)

#params
D_in,D_hidden,D_out = datax_train.shape[2],20,nb_cities

lr = 1e-5
n_epoch = 50

#model 
model = RNN(D_in,D_hidden,D_out,nn.functional.relu)
optimizer = torch.optim.Adam(model.parameters(),lr=lr)
loss_fn = torch.nn.CrossEntropyLoss()

#init
loss_train = []
acc_train = []
loss_test = []
acc_test = []

horizon = 1

#descent
for i in range(n_epoch):
    print("iteration {}".format(i))
    indices = torch.randperm(N)
    for j in range(N//batch_size):
        #batch
        ind = indices[j*batch_size:(j+1)*batch_size]
        x,y = datax_train[:,ind,:],datay_train[ind]
        #zero grad
        optimizer.zero_grad()
        #forward
        model.forward(x)
        yhat = model.decoder(horizon)
        #loss
        loss = loss_fn(yhat,y)
        #backpropagation
        loss.backward()
        #optimize
        optimizer.step()
        
    # Compute train loss and accuracy 
    model.forward(datax_train)
    Yhat = model.decoder(horizon)
    L = loss_fn(Yhat,datay_train)
    loss_train.append(L)
    _,indsYhat = torch.max(Yhat,dim=1)
    acc = torch.sum(datay_train == indsYhat,dtype=float)/len(Yhat)
    acc_train.append(acc)

##data train and test
length = 100
batch_size = 10
N = 5000
horizon = 3
#ind_cities = np.random.choice(list(range(len(data[0]))),size = nb_cities,replace = False)
ind_time = torch.randint(len(data)-length-1,size=(1,))
ind_city = torch.randint(len(ind_cities),size=(1,))
datax_train = torch.Tensor(data[ind_time:ind_time+length,ind_city])
datay_train = torch.Tensor(data[ind_time+length+1:ind_time+length+1+horizon,ind_city])
for i in range(N-1):
    ind_time = torch.randint(len(data)-length-1-horizon,size=(1,))
    datay_train = torch.cat((datay_train,torch.Tensor(data[ind_time+length+1:ind_time+length+1+horizon,ind_city])))
    datax_train = torch.cat((datax_train,torch.Tensor(data[ind_time:ind_time+length,ind_city])))
datay_train = datay_train.view(-1,horizon)
datax_train = datax_train.view(length,-1,1)

#params
D_in,D_hidden,D_out = datax_train.shape[2],20,1
lr = 1e-4
n_epoch = 80

#model 
model = RNN(D_in,D_hidden,D_out,torch.nn.functional.relu)
optimizer = torch.optim.Adam(model.parameters(),lr=lr)
loss_fn = torch.nn.MSELoss()

#init
loss_train = []
acc_train = []
loss_test = []
acc_test = []

#descent
for i in range(n_epoch):
    print("iteration {}".format(i))
    indices = torch.randperm(N)
    for j in range(N//batch_size):
        #batch
        ind = indices[j*batch_size:(j+1)*batch_size]
        x,y = datax_train[:,ind,:],datay_train[ind,:]
        #regression
        x = torch.cat((x,y.t().view(horizon,batch_size,1)),dim=0)
        #zero grad
        optimizer.zero_grad()
        #forward
        model.forward(x)
        yhat = model.decoder(horizon)
        #loss
        loss = loss_fn(yhat,y)
        #backpropagation
        loss.backward()
        #optimize
        optimizer.step()
        
    # Compute train loss and accuracy 
    model.forward(torch.cat((datax_train,datay_train.t().view(horizon,-1,1)),dim=0))
    Yhat = model.decoder(horizon)
    L = loss_fn(Yhat,datay_train)
    loss_train.append(L)


#Generative RNN 

import string
import unicodedata
LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0] = '' ##NULL CHARACTER
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

def normalize(s):
    return ''.join(c for c in unicodedata.normalize('NFD',s) if c in LETTRES)
def string2code(s):
    return torch.tensor([lettre2id[c] for c in normalize(s)])
def code2string(t):
    if type(t) != list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)

def oneshot(ind,dim):
    os = torch.zeros(len(ind), dim)
    for i in range(len(ind)):
        os[i,ind[i]] = 1
    return os

class RNN_GEN(nn.Module):
    def __init__(self,D_sym,D_emb,D_hidden,hidden_activ):
        super(RNN_GEN,self).__init__()
        self.linear_emb = nn.Linear(D_sym,D_emb)
        self.linear_hidden = nn.Linear(D_emb+D_hidden,D_hidden)
        self.linear_out = nn.Linear(D_hidden,D_sym)
        self.hidden_activ = hidden_activ
        self.list_decoded = []
        self.D_sym,self.D_emb,self.D_hidden = D_sym,D_emb,D_hidden
        
    def one_step(self,input):
        return self.hidden_activ(self.linear_hidden(torch.cat((input,self.h),dim=1)))
            
    def forward(self,x):
        self.list_decoded = torch.zeros((1,x.shape[1],self.D_sym))
        self.h = torch.zeros((x.shape[1],D_hidden))
        for i in range(len(x)):
            input = self.linear_emb(oneshot(x[i,:,:],self.D_sym))
            self.h = self.one_step(input)
            self.list_decoded = torch.cat((self.list_decoded,self.linear_out(self.h).view(1,x.shape[1],-1)))
    
    def generate(self,horizon,x):
        #one sequence at a time
        #not working yet to do again
        gen = []
        self.forward(x)
        sym = x[-1]
        for i in range(horizon):
            input = self.linear_emb(oneshot(sym,self.D_sym))
            self.h = self.one_step(input)
            _,sym = torch.max(self.linear_out(self.h),dim=1)
            gen.append(code2string(sym))
        return gen
                
#data
f = open("trump_full_speech.txt","r")
data = normalize(f.read())
f.close()
#encode the text
encoded = string2code(data)

length = 15
N = 5000
batch_size = 10
ind = torch.randint(len(data)-length,size=(1,))
datax = string2code(data[ind:ind+length]).view(-1,1,1)
for i in range(N-1):
    ind = torch.randint(len(data)-length,size=(1,))
    print(data[ind:ind+length])
    datax = torch.cat((datax,string2code(data[ind:ind+length]).view(-1,1,1)),dim=1)
    
#params
D_sym,D_emb,D_hidden = len(id2lettre),50,64
lr = 1e-4
n_epoch = 100
horizon = 5

#model 
model = RNN_GEN(D_sym,D_emb,D_hidden,nn.functional.relu)
optimizer = torch.optim.Adam(model.parameters(),lr=lr)
loss_fn = torch.nn.CrossEntropyLoss()

loss_train = []

#descent
for i in range(n_epoch):
    print("iteration {}".format(i))
    indices = torch.randperm(N)
    for j in range(N//batch_size):
        #batch
        ind = indices[j*batch_size:(j+1)*batch_size]
        x = datax[:,ind]
        #zero grad
        optimizer.zero_grad()
        #forward
        model.forward(x)
        yhat = model.list_decoded
        #loss
        loss = 0
        for k in range(1,length):
            loss += loss_fn(yhat[k],x[k].view(-1))
        loss/=(length-1)
        #backpropagation
        loss.backward()
        #optimize
        optimizer.step()
        
    # Compute train loss 
    model.forward(datax)
    yhat = model.list_decoded
    loss = 0
    for k in range(1,length):
        loss += loss_fn(yhat[k],datax[k].view(-1))
    loss/=(length-1)
    loss_train.append(loss.item())
        
print(model.generate(5,datax[:,0].view(1,-1,1)))
        
        
        
        




