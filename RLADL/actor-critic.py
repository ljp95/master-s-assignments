import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from utils import *
import numpy as np
import gym

class ActorCritic(nn.Module):
    def __init__(self,featExtractor,nb_features,action_space,gamma=0.99,device='cpu'):
        super(ActorCritic,self).__init__()
        self.featExtractor = featExtractor
        self.actor = NN(featExtractor.outSize,action_space.n,nb_features)
        self.critic = NN(featExtractor.outSize,1,nb_features)
        
        self.rewards = []
        self.values = []
        self.action_probas = []
        
        self.gamma = gamma
        self.action_space = action_space.n
        self.device = device
        self.train = True
    
    def forward(self,features,train=True):
        proba = F.softmax(self.actor(features),dim=0)
        if train:
            action = np.random.choice(self.action_space,p=proba.cpu().detach().numpy())
        else:
            action = torch.argmax(proba).item()
        value = model.critic(features)
        return action,proba[action],value
    
    def reinitialize_save(self):
        self.action_probas = []
        self.rewards = []
        self.values = []


def play_episode(model,env,T=200,train=True):   
    obs = env.reset()
    model.reinitialize_save()
    sum_reward = 0
    for t in range(T):
        features = torch.Tensor(model.featExtractor.getFeatures(obs)).to(model.device)
        action,action_proba,value = model(features,train)
        new_obs,reward,done,_ = env.step(action)
        
        sum_reward += reward
        model.rewards.append(reward)
        model.values.append(value)
        model.action_probas.append(action_proba)
        
        obs = new_obs
        if done:
            break    
    return sum_reward

def update(model,optimizer,criterion):
    if model.rewards == []:
        print("No rewards !")
        return 
    gt_returns = [0]
    loss = 0
    
    for reward in model.rewards[::-1]:
        gt_return = reward + model.gamma*gt_returns[-1]
        gt_returns.append(gt_return)
    gt_returns.pop(0)
    gt_returns = torch.tensor(gt_returns[::-1]).to(model.device) 
    gt_returns = (gt_returns - gt_returns.mean()) / gt_returns.std()
    
    for gt_return,action_proba,value in zip(gt_returns,model.action_probas,model.values): 
        advantage = gt_return - value
        loss -= advantage*torch.log(action_proba) 
        loss += criterion(gt_return,value.squeeze(0))
            
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()    
    model.loss = loss.item()
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = load_yaml('./configs/config_random_cartpole.yaml')
env = gym.make(config["env"])
opt = load_yaml('./configs/config_random_cartpole.yaml')

featExtractor = opt.featExtractor(env)
nb_features = [128]
model = ActorCritic(featExtractor,nb_features,env.action_space,device=device).to(device)

nb_episodes = 1000
lr = 1e-3
optimizer = optim.Adam(model.parameters(),lr=lr)
criterion = nn.SmoothL1Loss()
train = True

for episode in range(nb_episodes):
    if episode%10 == 0:
        reward = play_episode(model,env,train=False)
        if episode%5 == 0:
            print("Episode {} \n\t reward : {} ".format(episode,reward))
        
    else:
        reward = play_episode(model,env,train=True)
        
        update(model,optimizer,criterion)
        if episode%5 == 0:
            print("Episode {} \n\t reward : {} \n\t Loss : {}".format(episode,reward,model.loss))




