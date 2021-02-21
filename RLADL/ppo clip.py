import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Categorical
from utils import *
import gym

                
class ActorCritic(nn.Module):
    def __init__(self,featExtractor,nb_features,action_space,gamma=0.99,device='cpu',clip=0.2,K=5,actor_lr=1e-4,critic_lr=1e-3,update_timestep=2000):
        super(ActorCritic,self).__init__()
        self.featExtractor = featExtractor
        
        self.actor = NN(featExtractor.outSize,action_space,nb_features)
        self.new_actor = NN(featExtractor.outSize,action_space,nb_features)
        self.new_actor.load_state_dict(self.actor.state_dict())
        
        self.critic = NN(featExtractor.outSize,1,nb_features)
        
        self.gamma = gamma
        self.device = device
        self.clip = clip
        self.action_space = action_space
        self.K = K
        
        self.dones = []
        self.actions_probas = []
        self.rewards = []
        self.actions = []
        self.features = []
        self.values  = []
        
        self.actor_optimizer = optim.Adam(self.new_actor.parameters(),lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=critic_lr)
        self.timestep = 0
        self.update_timestep = update_timestep
    
    def select_action(self,features,random=False):
        probabilities = F.softmax(self.actor(features),dim=1)
        distributions = Categorical(probabilities)
        if random:
            actions = distributions.sample()
        else:
            actions = torch.argmax(probabilities,dim=1)
        self.actions.append(actions)        
        self.actions_probas.append(distributions.log_prob(actions))
        return actions.item()
        
    def evaluate(self,features,actions,new=True):
        if new:
            distributions = Categorical(F.softmax(self.new_actor(features),dim=0))
        else:
            distributions = Categorical(F.softmax(self.actor(features),dim=0))
        values = self.critic(features)
        return distributions.log_prob(actions),values
    
    def reinitialize_save(self):
        self.dones = []
        self.actions_probas = []
        self.rewards = []
        self.features = []
        self.actions = []
        self.values = []
        
def play_episode(model,env,T=200,random=True):   
    obs = env.reset()
    sum_reward = 0    
    for t in range(T):
        features = torch.Tensor(model.featExtractor.getFeatures(obs)).unsqueeze(0).to(model.device)
        action = model.select_action(features,random=random)
        new_obs,reward,done,_ = env.step(action)
        sum_reward += reward
        obs = new_obs
        if random:
            model.timestep += 1
            model.rewards.append(reward)
            model.features.append(features)
            model.dones.append(done)
        if done:
            break    
    return sum_reward

def update(model,criterion):
    ''' initializing '''
    gt_returns = []
    gt_return = 0
    device = model.device
    
    ''' computing and normalizing the returns '''
    for reward,done in zip(reversed(model.rewards),reversed(model.dones)):
        gt_return = reward + model.gamma*gt_return*(1-done)
        gt_returns.insert(0,gt_return)
    gt_returns = torch.tensor(gt_returns).float().to(device) 
    gt_returns = (gt_returns - gt_returns.mean()) / (gt_returns.std() + 1e-5)
    
    ''' converting lists to tensors '''
    old_actions = torch.stack(model.actions).to(device).detach()
    old_features = torch.stack(model.features).to(device).detach()
    old_actions_probas = torch.stack(model.actions_probas).to(device).detach()      
    
    ''' losses and backward '''
    for s in range(model.K):        
        log_probas,values = model.evaluate(old_features,old_actions,new=True)
        advantages = gt_returns - values.detach().view(-1)
        ratios = torch.exp(log_probas - old_actions_probas).view(-1)
        
        surrogate1 = torch.clamp(ratios,1-model.clip,1+model.clip)*advantages
        surrogate2 = ratios*advantages
        surrogate_loss = -torch.mean(torch.min(surrogate1,surrogate2))
        
        values_loss = criterion(values.view(-1),gt_returns)
        
        loss = surrogate_loss + values_loss
        model.actor_optimizer.zero_grad()
        model.critic_optimizer.zero_grad()
        loss.backward()
        model.actor_optimizer.step()   
        model.critic_optimizer.step()   

    ''' update the old model '''
    model.actor.load_state_dict(model.new_actor.state_dict())
    return loss.item()
        
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = load_yaml('./configs/config_random_cartpole.yaml')
opt = load_yaml('./configs/config_random_cartpole.yaml')

# config = load_yaml('./configs/config_random_lunar.yaml')
# opt = load_yaml('./configs/config_random_lunar.yaml')

env = gym.make(config["env"])
action_space = env.action_space.n
featExtractor = opt.featExtractor(env)
nb_features = [64,64]
model = ActorCritic(featExtractor,nb_features,action_space,device=device,update_timestep=128).to(device)

nb_episodes = 1000
criterion = nn.MSELoss()

for episode in range(nb_episodes):
    if episode%10 == 0:
        reward = play_episode(model,env,random=False)
        if episode%5 == 0:
            print("Episode {} \n\t reward : {} ".format(episode,reward))
        model.reinitialize_save()
            
    else:
        reward = play_episode(model,env,random=True)
        if model.timestep > model.update_timestep:
            loss = update(model,criterion)
            model.reinitialize_save()
            model.timestep = 0
            print("Episode {} \n\t reward : {} \n\t Loss : {:.2f}".format(episode,reward,loss))
            



