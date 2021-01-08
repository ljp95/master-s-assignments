
import gym
from gym import wrappers, logger
import gridworld2
import numpy as np
import matplotlib 
matplotlib.use("TkAgg")
from collections import defaultdict

def eGreedy(e,Q,s):
    if(np.random.uniform(0,1)<e):
        return np.random.randint(len(Q[s]))
    else:
        return np.argmax(Q[s])

''' init env'''
env = gym.make('gridworld-v1')
#outdir = 'gridworld-v1/random-agent-results'
#env = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
env.setPlan("gridworldPlans/plan5.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})

''' init values '''
qvalues = defaultdict(lambda : np.zeros(4))
rsum = 0
pause = 0.005
nb_actions = 4

''' hyperparameters '''
alpha = 0.001
gamma = 0.99
tau = 0.999
episode_count = 10000
eps = 0.05

for i in range(episode_count):
    state = env.reset()    
    if(i):
        tau *= 0.999
    if(tau<=0.01):
        print('Temperature too small, stopping ... \n End at episode : {}'.format(str(i)))
        break
    
    rsum,reward,j = 0,0,0
    env.verbose = (i % 100 == 0 and i > 0) 
    if env.verbose:
        env.render(pause)
        
    action = eGreedy(eps,qvalues,state)    
    
    while(True):
        j += 1
        next_state,reward,done,_ = env.step(action)
        rsum += reward
        next_action = eGreedy(eps,qvalues,next_state)        
        qvalues[state][action] += alpha * (reward + gamma*qvalues[next_state][next_action] - qvalues[state][action])
        state = next_state
        action = next_action
        if env.verbose:
            env.render(pause)
        if(done):
            print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
            break
print("done")
env.close()

