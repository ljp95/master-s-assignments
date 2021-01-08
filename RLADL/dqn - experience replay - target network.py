import argparse
import sys
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use("Qt5ag")
matplotlib.use("TkAgg")
import gym
#import gridworld
import torch
from utils import *
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import memory
from collections import defaultdict
import copy
import torch.optim as optim
import time

def eGreedy(eps,decay,Q,s):
    threshold = eps*decay
    if(torch.rand(1)<threshold):
        return threshold,torch.randint(high=len(Q(torch.tensor(s))),size=(1,)).item()
    else:
        return threshold,torch.argmax(Q(torch.tensor(s))).item()

device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"

''' init env'''
config = load_yaml('./configs/config_random_cartpole.yaml')
env = gym.make(config["env"])
agent = RandomAgent(env,config)
opt = load_yaml('./configs/config_random_cartpole.yaml')

''' hyperparameters '''
nb_features = 256
N = 100000
batch_size = 2500
T = 500
nb_episode = 1000
iteration = 1
rb_min = 3000
lr = 0.001
C = 150
eps = 0.75
decay = 0.99998
mu = 0.01
gamma = 0.999
list_reward = []
test = False

''' init '''
action_space = env.action_space
featureExtractor = opt.featExtractor(env)
Q = NN(featureExtractor.outSize,action_space.n,[nb_features]).double().to(device)
Qtarget = copy.deepcopy(Q).to(device)
D = memory.Memory(N)
optimizer = optim.Adam(Q.parameters(),lr=lr)
criterion = nn.SmoothL1Loss()

for episode in range(nb_episode):
    obs = env.reset()
    features = agent.featureExtractor.getFeatures(obs)
    rsum = 0
    
    debut = time.time()
    for t in range(1,T+1):
        Q = Q.to("cpu")
        if test:
            action = np.argmax(Q(torch.from_numpy(features)).detach().numpy())
        else:   
            eps,action = eGreedy(eps,decay,Q,features)
            
        new_obs,reward,done,_ = env.step(action)
        new_features = featureExtractor.getFeatures(new_obs)
        D.store((features,action,reward,new_features,done))
        rsum += reward
        
        if not test:
            if D.tree.nentities > rb_min:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                Q = Q.to(device)
                Qtarget = Qtarget.to(device)
                batch = np.vstack(D.sample(batch_size)[2])
                batch_features = torch.tensor(np.vstack(batch[:,0]).astype('float32')).to(device)
                batch_action = torch.tensor(batch[:,1].astype('int64')).to(device)
                batch_reward = torch.tensor(batch[:,2].astype('float32')).to(device)
                batch_new_features = torch.tensor(np.vstack(batch[:,3]).astype('float32')).to(device)
                batch_done = torch.tensor((batch[:,4]).astype("bool")).to(device)*1
                batch_Q = Q(batch_features.double())[torch.arange(len(batch_action)),batch_action].to(device)
                
                with torch.no_grad():
                    y = batch_reward + (1-batch_done) * gamma * torch.max(Qtarget(batch_new_features.double()),dim=1)[0]
                    
                loss = criterion(y.double(),batch_Q.double())
                iteration += 1
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if iteration%C == 0:
                    Qtarget = copy.deepcopy(Q)
                                
        features = new_features
        
        if episode%10 == 0:
            test = True
        else:
            test = False
        if done:
            if test:
                print(f"Cumulative reward in test : {rsum}")
                list_reward.append(rsum)

            elif D.tree.nentities > rb_min:
                print(f"Loss : {loss} /// Cumulative reward : {rsum} in {round((time.time()-debut),2)}s")        
            break
            
