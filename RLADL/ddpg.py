import torch
import torch.optim as optim
import torch.nn as nn
from utils import *
import gym
import random
from collections import namedtuple
import matplotlib.pyplot as plt

Transition = namedtuple('Transition',('features', 'action', 'reward', 'done', 'new_features'))

class Buffer:
    def __init__(self,size):
        self.size = size
        self.datas = []
        
    def store(self,*args):
        if len(self.datas) == self.size:
            self.datas.pop(0)
        self.datas.append(Transition(*args))
        
    def sample_batch(self,batch_size):
        return random.sample(self.datas,batch_size)
        
    def reinitialize(self):
        self.datas = []
        
    def len(self):
        return len(self.datas)
                
class DDPG(nn.Module):
    def __init__(self,input_size,nb_features,action_space,gamma=0.99,device='cpu',criterion=nn.MSELoss()):
        super(DDPG,self).__init__()
        self.actor = NN_bn(input_size,action_space.shape[0],nb_features)
        self.new_actor = NN_bn(input_size,action_space.shape[0],nb_features)
        self.new_actor.load_state_dict(self.actor.state_dict())
        self.critic = critic_bn(input_size,action_space.shape[0],nb_features)
        self.new_critic = critic_bn(input_size,action_space.shape[0],nb_features)
        self.new_critic.load_state_dict(self.critic.state_dict())
        
        self.gamma = gamma
        self.device = device
        self.action_space = action_space
        
        self.criterion = criterion
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=1e-3)
        self.gaussian_factor = 1
    
    def select_action(self,features,noise=True):
        self.actor.eval()
        action = self.actor(features).data
        self.actor.train()
        if noise:
            action += (torch.randn(len(action))*self.gaussian_factor).to(self.device)
        action = torch.clamp(action,self.action_space.low[0],self.action_space.high[0])
        return action
    
def play_episode(model,env,T=10000,noise=True,buffer=None):   
    obs = env.reset()
    sum_reward = 0    
    features = torch.Tensor([obs]).to(model.device)
    
    for t in range(T):
        action = model.select_action(features,noise)
        new_obs,reward,done,_ = env.step(action[0].detach().cpu().numpy())
        
        new_features = torch.Tensor([new_obs]).to(model.device)            
        
        if buffer:
            if done and t==998:
                done = False
            buffer.store(features,action,reward,done,new_features)  
            
        sum_reward += reward
        features = new_features

        if done:
            break    
    if buffer:
        model.gaussian_factor *= 0.995          
    return sum_reward

def update(model,buffer,batch_size):
    transitions = buffer.sample_batch(batch_size)
    batch = Transition(*zip(*transitions))
    batch_features = torch.cat(batch.features)
    batch_action = torch.cat(batch.action)
    batch_reward = torch.tensor(batch.reward).unsqueeze(1).to(model.device)
    batch_done = torch.tensor(batch.done).unsqueeze(1).to(model.device)
    batch_new_features = torch.cat(batch.new_features)
    
    new_q_values = model.new_critic(batch_new_features,model.new_actor(batch_new_features).detach())        
    q_targets = batch_reward + model.gamma*(~batch_done)*new_q_values
    q_values_noisy = model.critic(batch_features,batch_action)
    
    q_targets_loss = model.criterion(q_values_noisy,q_targets.detach())
    model.critic_optimizer.zero_grad()
    q_targets_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.critic.parameters(), 0.5)    
    model.critic_optimizer.step()
    
    q_values = model.critic(batch_features,model.actor(batch_features))
    policy_loss = -torch.mean(q_values)
    model.actor_optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 0.5)    
    model.actor_optimizer.step()
    
    soft_update(model.new_actor,model.actor)
    soft_update(model.new_critic,model.critic)
    
    return (q_targets_loss+policy_loss).item()

def soft_update(target,source,tau=0.001):
    for target_param, source_param in zip(target.parameters(),source.parameters()):
        target_param.data.copy_(target_param.data*(1-tau) + source_param.data*tau)
      
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu' #faster on cpu
env = gym.make('MountainCarContinuous-v0')
# env = gym.make('LunarLanderContinuous-v2')
# env = gym.make('Pendulum-v0')

nb_features = [64,64]
criterion = nn.MSELoss()
model = DDPG(env.observation_space.shape[0],nb_features,env.action_space,device=device,criterion=criterion).to(device)

nb_episodes = 400
min_size_to_update = 1000
buffer = Buffer(50000)
batch_size = 64
rewards_test = []
rewards_train = []

for episode in range(nb_episodes):
    if episode%10 == 0:
        reward = play_episode(model,env,noise=False,buffer=None)
        rewards_test.append(reward)
        print("Episode {} \n\t reward : {:.2f}\n".format(episode,reward))
            
    else:
        reward = play_episode(model,env,noise=True,buffer=buffer)
        rewards_train.append(reward)
        if buffer.len() >= min_size_to_update:
            loss = update(model,buffer,batch_size)
            # print("Episode {} \n\t reward : {:.2f}\n".format(episode,reward))
            print("Episode {} \n\t reward : \t{:.2f} \n\t Loss : \t{:.2f}\n".format(episode,reward,loss))

plt.figure()
plt.plot(range(len(rewards_test)),rewards_test)
plt.title("test")
plt.figure()
plt.plot(range(len(rewards_train)),rewards_train)
plt.title("train")