
import gym
from gym import wrappers, logger
import gridworld2
import numpy as np
import matplotlib 
matplotlib.use("TkAgg")
from collections import defaultdict

''' init env'''
env = gym.make('gridworld-v1')
outdir = 'gridworld-v1/random-agent-results'
env = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
env.setPlan("gridworldPlans/plan9.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})

''' init values '''
qvalues = defaultdict(lambda : np.zeros(4))
rhat = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : 0)))
phat = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : 0)))

rsum = 0
pause = 0.00000001
nb_actions = len(P[list(P.keys())[0]])

''' hyperparameters '''
alpha = 0.0005
alphar = 0.0005
gamma = 0.99
tau = 0.99
n = 50
episode_count = 4000

for i in range(episode_count):
    state = env.reset()    
    if(i):
        tau *= 0.999
    if(tau<=0.01):
        print('Temperature too small, have to stop \n End at episode : {}'.format(str(i)))
        break
    
    rsum,reward,j = 0,0,0
    env.verbose = (i % 1 == 0 and i > 0) 
    if env.verbose:
        env.render(pause)
        
    while(True):
        current_reward = reward
        j += 1
        
        #qlearning
        proba = np.exp(np.array(qvalues[state])/tau)
        proba = proba/np.sum(proba)
        action = np.random.choice(nb_actions,p=proba)
        new_state,reward,done,_ = env.step(action)
        rsum += reward
        qvalues[state][action] += alpha*(reward+gamma*np.max(qvalues[new_state])-qvalues[state][action])
        
        #maj
        rhat[state][action][new_state] += alphar*(reward-rhat[state][action][new_state])
        phat[state][action][new_state] += alphar*(reward-phat[state][action][new_state])
        for state_prime in set(phat[state][action].keys())-set({new_state}):
            phat[state][action][state_prime] -= alphar*phat[state][action][state_prime]
            
        for k in range(n):
            random_state,random_action = np.random.choice(len(qvalues)),np.random.choice(nb_actions)
            sum_rhat = 0
            for state_prime in set(phat[state][action].keys())-set({new_state}):
                sum_rhat += phat[random_state][random_action][state_prime]*(rhat[random_state][random_action][state_prime]+gamma*np.max(qvalues[state_prime]))-qvalues[random_state][random_action]
            qvalues[random_state][random_action] += alpha*(sum_rhat)
        
        state = new_state
        if env.verbose:
            env.render(pause)
        if(done):
            print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
           
            break
print("done")
env.close()




