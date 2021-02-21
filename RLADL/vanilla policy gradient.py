import torch
import torch.optim as optim
import torch.nn.functional as F
from utils import *
import numpy as np
import gym

class PolicyGradient(nn.Module):
    def __init__(self,featExtractor,nb_features,action_space,gamma=0.99,device='cpu'):
        super(PolicyGradient,self).__init__()
        self.featExtractor = featExtractor
        self.actor = NN(featExtractor.outSize,action_space.n,nb_features)
        self.rewards = []
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
        return action,proba[action]
    
    def reinitialize_save(self):
        self.action_probas = []
        self.rewards = []
    
def play_episode(model,env,T=200,train=True):   
    obs = env.reset()
    model.reinitialize_save()
    sum_reward = 0
    for t in range(T):
        features = torch.Tensor(model.featExtractor.getFeatures(obs)).to(model.device)
        action,action_proba = model(features,train)
        new_obs,reward,done,_ = env.step(action)
        sum_reward += reward
        model.rewards.append(reward)
        model.action_probas.append(action_proba)
        obs = new_obs
        if done:
            break    
    return sum_reward

def update(model,optimizer):
    if model.rewards == []:
        print("No rewards !")
        return 
    
    gt_returns = [0]
    loss = 0
    for reward in model.rewards[::-1]:
        gt_return = reward + model.gamma*gt_returns[-1]
        gt_returns.append(gt_return)
    gt_returns = torch.tensor(gt_returns[::-1]).to(model.device) 
    gt_returns = (gt_returns - gt_returns.mean()) / gt_returns.std()
    
    for gt_return,action_proba in zip(gt_returns,model.action_probas):    
        loss -= gt_return*torch.log(action_proba)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()    
    model.loss = loss
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = load_yaml('./configs/config_random_cartpole.yaml')
env = gym.make(config["env"])
opt = load_yaml('./configs/config_random_cartpole.yaml')

featExtractor = opt.featExtractor(env)
nb_features = [128]
model = PolicyGradient(featExtractor,nb_features,env.action_space,device=device).to(device)

nb_episodes = 1000
lr = 1e-3
optimizer = optim.Adam(model.parameters(),lr=lr)
train = True

for episode in range(nb_episodes):
    if episode%10 == 0:
        reward = play_episode(model,env,train=False)
        if episode%5 == 0:
            print("Episode {} \n\t reward : \t{} ".format(episode,reward))
        
    else:
        reward = play_episode(model,env,train=True)
        
        update(model,optimizer)
        if episode%5 == 0:
            print("Episode {} \n\t reward : \t{} \n\t Loss : \t{}".format(episode,reward,model.loss))




