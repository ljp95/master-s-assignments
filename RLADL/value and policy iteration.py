
import matplotlib 
matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy

def ValueIteration(env, gamma=0.99, eps=1e-6, diff=1e-8):
    states,P = env.getMDP()
    random_key = list(P.keys())[0]
    nb_actions = len(P[random_key])
    V = np.zeros((len(states)))
    
    while(diff>eps):
        tmp_V = copy.deepcopy(V)
        for state in P.keys():
            tmp_values = np.zeros((nb_actions))
            state_nb = states[state]
            for action in range(nb_actions):
                for new_state_nb in range(len(P[state][action])):
                    proba,obs,reward,done = P[state][action][new_state_nb] 
                    obs_nb = states[obs]
                    tmp_values[action] += proba*(reward + gamma*tmp_V[obs_nb]) 
            V[state_nb] = np.max(tmp_values) 
        diff = np.linalg.norm((V-tmp_V))
    diff = eps*2
    
    policy = np.zeros_like(V)
    
    for state in P.keys():
        tmp_values = np.zeros((nb_actions))
        state_nb = states[state]
        for action in range(nb_actions):
            for new_state_nb in range(len(P[state][action])):
                proba,obs,reward,done = P[state][action][new_state_nb] 
                obs_nb = states[obs]
                tmp_values[action] += proba*(reward + gamma*V[obs_nb]) 
        policy[state_nb] = np.argmax(tmp_values)
        
    return policy

def PolicyIteration(env, gamma=0.99, eps=1e-6, diff=1e-8):
    states,P = env.getMDP()
    random_key = list(P.keys())[0]
    nb_actions = len(P[random_key])
    policy = np.zeros((len(states)))
    tmp_policy = np.zeros_like(policy)
    
    while(True):
        V = np.zeros((len(states)))
        while(diff>eps):
            tmp_V = copy.deepcopy(V)
            V = np.zeros((len(states)))
            for state in P.keys():
                state_nb = states[state]
                action = policy[state_nb]
                for index_choice in range(len(P[state][action])):
                    proba,obs,reward,done = P[state][action][index_choice] 
                    obs_nb = states[obs]
                    V[state_nb] += proba*(reward + gamma*tmp_V[obs_nb]) 
            diff = max(np.abs(V-tmp_V))
        diff = eps*2    
            
        tmp_policy = copy.deepcopy(policy)
        
        for state in P.keys():
            tmp_values = np.zeros((nb_actions))
            state_nb = states[state]
            for action in range(nb_actions):
                for index_choice in range(len(P[state][action])):
                    proba,obs,reward,done = P[state][action][index_choice] 
                    obs_nb = states[obs]
                    tmp_values[action] += proba*(reward + gamma*V[obs_nb]) 
            policy[state_nb] = np.argmax(tmp_values)
            
        if(np.all(policy==tmp_policy)):
            print("breaking with {}".format(V))
            break
        
    return policy



class Agent(object):
    def __init__(self,policy,states):
        self.policy = policy
        self.states = states

    def act(self,observation):
        return self.policy[self.states[gridworld.GridworldEnv.state2str(observation)]]
        
if __name__ == '__main__':
    env = gym.make('gridworld-v0')
    env.seed(0)
    env.render(mode="human") 
    
    print("Learning ...")
    policy = ValueIteration(env)
    policy = PolicyIteration(env,gamma = 0.5)
    states,P = env.getMDP()
    agent = Agent(policy,states)

#    outdir = 'gridworld-v0/agent-results'
#    env = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    
    episode_count = 5
    reward = 0
    done = False
    rsum = 0
    pause = 0.00001
    
    for i in range(episode_count):
        obs = env.reset()
        env.verbose = (i%1 == 0)
        if env.verbose:
            env.render(pause)
        j = 0
        rsum = 0
        while True:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            rsum += reward
            j += 1
            if env.verbose:
                env.render(pause)
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("end")
    env.close()