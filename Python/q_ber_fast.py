# import math
import time
import random
import numpy as np
# import matplotlib.pyplot as plt

ns = 5000 #number of steps per episode
ne = 1 #number of episodes
e = 10 #probability taking a random action
delta = 0.9 #discount factor
l_rate = 0.8 #learning rate
r_list = [0.3, 0.5] #list of the server rates
l = 0.5 #lambda
d = 2 #d
max_queue_length = 5 #max_queue_length
save = 5 #number of saved qtables

class MDP():

    def __init__(self, r_list, Lambda, max_queue_lenght):
        self.LAMBDA = Lambda
        self.R = r_list
        self.N = len(r_list)
        self.state = [0]*self.N
        self.max = max_queue_lenght
    
    def reset(self):
        self.state = [0]*self.N
        return [self.state, 0]

    def action(self, action):
        cost = 0

        #The new job is routed according to the probabilities of the action
        routed_q = random.choices([i for i in range(self.N)], weights = action[0])[0]
        self.state[routed_q] += 1

        for i in range(self.N):
            size = self.state[i]
            
            #the size of the queue is added to the cost
            cost += - size

            #The queues sizes are updated
            if size != 0:
                served_jobs = random.choices([k for k in range(size + 1)], weights=[self.__q(i,j) for j in range(size + 1)])[0]
                self.state[i] -= served_jobs

        #make state for return considering maximum queue size      
        state = [0 for _ in range(self.N)]
        index = 0
        for queue in range(self.N):
            state[queue] = self.state[queue]
            if self.state[queue] > self.max:
                state[queue] = self.max
            index += state[queue]*(self.max + 1)**(self.N - queue -1)

        return (state, cost)

    def state(self):
        return self.state

    def __q(self, i, j): #the probability that j jobs are served at Queue i
        if j == self.state[i]:
            return((self.R[i] / (self.R[i] + self.LAMBDA))**j)
        else:
           return(self.LAMBDA * (self.R[i]**j) / (self.R[i] + self.LAMBDA)**j)

def rec(action, d, N, sum, len):
    if len > N:
        return []
    elif len == N:
        if sum == d:
            return [action]
        return []
    elif sum == d:
        return rec(action + [0], d, N, sum, len+1)
    ans = []
    i = 0
    while sum+i <= d:
        ans += rec(action+[i/d], d, N, sum+i, len+1)
        i += 1
    return ans

def action_space(d, N):
    a = rec([], d, N, 0, 0)
    return [[a[i], i] for i in range(len(a))]

def rec_s(state, N, len, max_queue_length):
    if len > N:
        return []
    elif len == N:
        return [state]
    ans = []
    i = 0
    while i <= max_queue_length:
        ans += rec_s(state+[i], N, len+1, max_queue_length)
        i += 1
    return ans

def state_space(N, max):
    s = rec_s([], N, 0, max)
    return [[s[i], i] for i in range(len(s))]

def qber(ns, ne, e, delta, l_rate, r_list, l, d, max_queue_length, save):

    N = len(r_list)
    #generation of the action space
    action_s = action_space(d, N)
    action_s_size = len(action_s)

    #generation of the states
    state_s = state_space(N, max_queue_length)
    state_s_size = len(state_s)

    #generation of the q-table
    q = np.zeros((state_s_size, action_s_size))

    mdp = MDP(r_list, l, max_queue_length)

    for episode in range(ne ):
        
        #get initial state
        state = mdp.reset()

        for step in range(1, ns+1): #starts on 1 to avoid /0 in learning rate
            
            if random.uniform(0,1) > e: #take minimun quality action
                action_number = np.argmax(q[state[1]])
                action = action_s[action_number]
            else: #take random action
                action = random.choices(action_s)[0]

            #take action, get new_state and reward
            (new_state, cost) = mdp.action(action)
            
            #update q-table
            q[state[1], action[1]] = q[state[1], action[1]] + (1/(step**l_rate))*(cost + delta*np.max(q[new_state[1], :]) - q[state[1], action[1]])

            state = new_state
            e -= 0.0001

            if ns - step < save:
                np.save('Python/' + str(save - ns + step)+ 'q', q)
    
    return q

t0 = time.time()

q = qber(ns, ne, e, delta, l_rate, r_list, l, d, max_queue_length, save)
# np.save('Python/q', q)

print(time.time()-t0)