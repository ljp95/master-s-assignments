
import gym

import multiagent
import multiagent.scenarios
import multiagent.scenarios.simple_tag as simple_tag
import multiagent.scenarios.simple_tag as simple_spread
import multiagent.scenarios.simple_tag as simple_adversary
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from gym import wrappers, logger
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import namedtuple
import matplotlib.pyplot as plt
from utils import *


"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.
Can be called by using, for example:
    env = make_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym
environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""

def make_env(scenario_name, benchmark=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    world.dim_c = 0
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    env.discrete_action_space = False
    env.discrete_action_input = False
    scenario.reset_world(world)
    return env,scenario,world

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

class ActorCritic(nn.Module):
    def __init__(self,stateSize,nb_features,actionSize,nb_agents,device='cpu'):
        super(ActorCritic,self).__init__()
        
        self.actor = NN_bn(stateSize,actionSize,nb_features)
        self.new_actor = NN_bn(stateSize,actionSize,nb_features)
        self.new_actor.load_state_dict(self.actor.state_dict())
        
        self.critic = critic_bn(stateSize,actionSize,nb_features)
        self.new_critic = critic_bn(stateSize,actionSize,nb_features)
        self.new_critic.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=1e-2)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=1e-2)
        
        self.actionSize = actionSize
        self.device = device
        
    def select_action(self,features,gaussian_factor=0.2,noise=True):
        self.actor.eval()
        action = self.actor(features).data
        self.actor.train()
        if noise:
            action += (torch.randn(action.shape[-1])*gaussian_factor).to(self.device)
        action = torch.clamp(action,-1,1) 
        return action 
        
class MADDPG(nn.Module):
    def __init__(self,nb_agents,stateSize,nb_features,actionSize,gamma=0.95,device='cpu',criterion=nn.MSELoss()):
        super(MADDPG,self).__init__()
        self.nb_agents = nb_agents
        self.agents = [ActorCritic(stateSize,nb_features,actionSize,nb_agents,device) for i in range(nb_agents)]
        self.gamma = gamma
        self.device = device
        self.criterion = criterion
        self.gaussian_factor = 0.3

    def select_actions(self,features,noise=True):
        return [model.agents[i].select_action(features[i],model.gaussian_factor,noise) for i in range(model.nb_agents)]
    
def soft_update(target,source,tau=0.01):
    for target_param, source_param in zip(target.parameters(),source.parameters()):
        target_param.data.copy_(target_param.data*(1-tau) + source_param.data*tau)
      
def obs_to_tensor(obs):
    return [torch.tensor(obs[i]).unsqueeze(0).float() for i in range(len(obs))]

def actions_to_numpy(actions):
    return [actions[i].squeeze(0).numpy() for i in range(len(actions))]

def play_episode(model,env,T=25,noise=False,buffer=None):
    obs = env.reset()
    for t in range(T):
        actions = model.select_actions(obs_to_tensor(obs),noise)
        new_obs,rewards,dones,_ = env.step(actions_to_numpy(actions))
        rewards = rewards[0]
        if buffer:
            buffer.store(torch.tensor(obs).unsqueeze(0),torch.cat(actions).unsqueeze(0),rewards,dones,torch.tensor(new_obs).unsqueeze(0))
        obs = new_obs
        if sum(dones)==len(env.agents):
            break
        if env.verbose:
            time.sleep(0.1)        
            env.render()   
    if buffer:
        model.gaussian_factor *= 0.9999     

    return rewards
        
def update(model,buffer,batch_size):
    sum_q_targets_loss = 0
    sum_policy_loss = 0
    nb_agents = model.nb_agents
    
    for i in range(model.nb_agents):
        agent = model.agents[i]
        
        transitions = buffer.sample_batch(batch_size)
        batch = Transition(*zip(*transitions))
        batch_features = torch.cat(batch.features,dim=0).float()
        batch_action = torch.cat(batch.action).reshape(batch_size,-1)
        batch_reward = torch.tensor(batch.reward).to(model.device).unsqueeze(1)
        batch_done = torch.tensor(batch.done).to(model.device)[:,i].unsqueeze(1)
        batch_new_features = torch.cat(batch.new_features,dim=0).float()
        
        new_actions = torch.cat([F.softmax(model.agents[i].new_actor(batch_new_features[:,i]),dim=1).unsqueeze(0) for i in range(nb_agents)]).permute(1,0,2).reshape(batch_size,-1)
        new_q_values = agent.new_critic(batch_new_features.reshape(batch_size,-1),new_actions.detach())        
        q_targets = batch_reward + model.gamma*(~batch_done)*new_q_values
        q_values_noisy = agent.critic(batch_features.reshape(batch_size,-1),batch_action)
        
        q_targets_loss = model.criterion(q_values_noisy.float(),q_targets.detach().float())
        agent.critic_optimizer.zero_grad()
        q_targets_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)    
        agent.critic_optimizer.step()
        
        actions = torch.cat([F.softmax(model.agents[i].actor(batch_features[:,i]),dim=1).unsqueeze(0) for i in range(nb_agents)]).permute(1,0,2).reshape(batch_size,-1)
        q_values = agent.critic(batch_features.reshape(batch_size,-1),actions)
        policy_loss = -torch.mean(q_values)
        agent.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)    
        agent.actor_optimizer.step()
        
        sum_q_targets_loss += q_targets_loss
        sum_policy_loss += policy_loss
        
    for j in range(model.nb_agents):
        soft_update(model.agents[j].new_actor,model.agents[j].actor)
        soft_update(model.agents[j].new_critic,model.agents[j].critic)
    # print(sum_policy_loss,sum_q_targets_loss)
    return (sum_q_targets_loss + sum_policy_loss).item()

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu' #faster on cpu

env,scenario,world = make_env('simple_spread')
criterion = nn.MSELoss()
nb_agents = 3
obs = env.reset()
stateSize = len(obs[0])
nb_features = [64,64]
actionSize = 2
model = MADDPG(nb_agents,stateSize,nb_features,actionSize,device=device,criterion=criterion).to(device)

nb_episodes = 10000
min_size_to_update = 1024
buffer = Buffer(1e6)
batch_size = 1024
rewards_test = []
rewards_train = []

import matplotlib
matplotlib.use("TkAgg")
import time

for episode in range(nb_episodes):
    env.verbose = (episode % 10 == 0 and episode > 0) 
    env.verbose = False
    if episode%10 == 0:
        reward = play_episode(model,env,noise=False,buffer=None)
        rewards_test.append(reward)
        print("Episode {} \n\t reward : {:.2f}\n".format(episode,reward))
            
    else:
        reward = play_episode(model,env,noise=True,buffer=buffer)
        rewards_train.append(reward)
        if episode%4 == 0:
        
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

