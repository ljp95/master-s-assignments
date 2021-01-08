
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
env.setPlan("gridworldPlans/plan5.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})

''' init values '''
qvalues = defaultdict(lambda : np.zeros(4))
rsum = 0
pause = 0.01
nb_actions = 4

''' hyperparameters '''
alpha = 0.001
gamma = 0.99
tau = 0.999
episode_count = 10000

for i in range(episode_count):
    current_state = env.reset()    
    if(i):
        tau *= 0.999
    if(tau<=0.01):
        print('Temperature too small, have to stop \n End at episode : {}'.format(str(i)))
        break
    
    rsum,reward,j = 0,0,0
    env.verbose = (i % 100 == 0 and i > 0) 
    env.verbose = False
    if env.verbose:
        env.render(pause)
        
    while(True):
        current_reward = reward
        j += 1
        proba = np.exp(np.array(qvalues[current_state])/tau)
        proba = proba/np.sum(proba)
        action = np.random.choice(nb_actions,p=proba)
        obs,reward,done,_ = env.step(action)
        rsum += reward
        
        qvalues[current_state][action] += alpha * (reward + gamma*np.max(qvalues[obs]) - qvalues[current_state][action])
        current_state = obs
        if env.verbose:
            env.render(pause)
        if(done):
            print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
            break
print("done")
env.close()

