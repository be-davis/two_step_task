import random
import numpy as np
from matplotlib import pyplot as plt


states = [0,1,2]
stages = [1,2]
actions = [0,1]
s1_state = [0]
s2_states = [1,2]
N = 150 #number of trials
lr = 0.1
lam = 0.5
epsilon = 0.2


#returns a list a probabilites according to random selection#
def RandomWalk():
    x = 0
    y = np.round(np.random.uniform(.25,.75), decimals = 3)
    trial = [x]
    prob = [y]
    for i in range(1, N+1):
        move = np.random.uniform(-.5,.5)
        if move<0:
            x+=1
            y-=.025
        if move >= 0:
            x+=1
            y+=.025

        trial.append(x)
        prob.append(y)
    return prob


def mf():
    sars = [] #holds--> list of states, actions, and reward received for each trial
    
    #random walk lists for probabiliteis of....
    s1a0 = RandomWalk() #being in state 1 and choosing action 0
    s1a1 = RandomWalk() #being in state 1 and choosing action 1
    s2a0 = RandomWalk() #being in state 2 and choosing action 0
    s2a1 = RandomWalk() #being in state 2 and choosing action 1
        #example: calling s1a0[t] would return the probability of receceiving a reward
            #for stte 1 and action zero on trial t
            #updates for every trial up to N
    
    ## below are lists that hold booleans used for calculating stay probabilities according to
        #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3077926/pdf/nihms280176.pdf
            #Figure 2 (page 20)
    common_rewarded = []
    rare_rewarded = []
    common_unrewarded = []
    rare_unrewarded = []
    
    ### CURRENT PROBLEM: The bar plots are not coming out like the should for figure 2A
        ## The rewarded should always be higher than unrewarded. Mine are not showing that ###

    
    #below is my SARS(lambda) TD learning code
    #The math behind this algorithm for the specific task is best explained in the paper below
            #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5001643/pdf/pcbi.1005090.pdf
                #pages 4 and 5
    
    #empty q table
    qs = np.zeros((len(states), len(actions)))

    for t in range(0, N):
        state_list = [0] #list of states encountered in each trial-->1st state is always state zero (which is stage 1)
        action_list = [] #list of actions encountered in each trial --> 2 actions each trial
        trace  = [0] #eligibiltiy trace
        done = False
        score = 0
        state = 0 #starting state for each trial
        dt = np.zeros(2) ## reward prediction errors (RPE) for both stages
            #dt[0] --> stage 1 RPE
            #dt[1] --> stage 2 RPE
        
        tr = 0.7 ##common transition from stage 1 (state 0) to stage 2 (states 1 or 2)
         
        #adjusts probability of receing reward for each stage 2 state/action for each trial
        s2_probs = [s1a0[t], s1a1[t], s2a0[t], s2a1[t]] #different s2_probs for every trial
            #paper does this according to a gaussian random walk
                               
        #selecting first action for every trial
        if random.uniform(0,1)<epsilon:
            # explore action space: take random
            action = np.random.choice(actions)
        else:                
            # check learned q values & take the best action that maximizes reward (greedy!)
            action=np.argmax(qs[state])


        while not done:
                                
            if state in s1_state: #or if state == 0, same thing
                
                action_list.append(action)
                
                rand = random.random()
                if state == 0 and action == 0:
                    
                    if rand < tr:
                        next_state = 2
                        reward = 0
                        done = False
                    else:
                        next_state = 1
                        reward = 0
                        done = False
                elif state == 0 and action == 1:
                    if random.random() < tr:
                        next_state = 1
                        reward = 0
                        done = False
                    else:
                        next_state = 2
                        reward = 0
                        done = False
                
                #add next state to the list
                state_list.append(next_state)

                #select best action at that next state
                next_act = np.random.choice(actions) if random.random()<epsilon else np.argmax(qs[next_state])
                
                #add next action to the list
                action_list.append(next_act)

                #eligbility trace explained in paper above (page 4)
                trace.append(1)
            elif state in s2_states:
                rand = random.random()
                if state == 1 and action == 0:
                    if rand < s2_probs[0]:
                        reward = 10
                    else:
                        reward = 0
                elif state == 1 and action == 1:
                    if rand < s2_probs[1]:
                        reward = 10
                    else:
                        reward = 0
                elif state == 2 and action == 0:
                    if random.random() < s2_probs[2]:
                        reward = 10
                    else:
                        reward = 0
                elif state == 2 and action == 1:
                    if random.random() < s2_probs[3]:
                        reward = 10
                    else:
                        reward = 0
                
                trace.append(trace[-1] +1)
                
                
                ##all Q values only get updated in the SECOND stage (so when agent is in state 1 or 2)
                
                #RPE and q update for stage 1
                dt[0] = qs[state][action] - qs[0, action_list[0]]
                
                qs[0,action_list[0]] =  qs[state_list[0],action_list[0]] + lam*lr*dt[0]*trace[1]
                
                ##RPE and q update for stage 2
                dt[1] = reward - qs[state][action]
                qs[state,action] = qs[state, action] + lr*dt[1]*trace[2]
                
                done = True
                
                
                #action_list.append(action)
            

            state = next_state
            action = next_act
            score+= reward
        sars.append([state_list])
        sars[-1].append(action_list)
        sars[-1].append([score])
    
    
    ##calulating stay probabilites:
    ##:sars[trial][0=state_list, 1=action_list, 2=reward][element in that list]
        #example: sars[20][0][0]--> the first state on episode 20 (always zero!)
        #example: sars[20][0][1]--> second state on episode 20 (either 1 or 2)
        #example: sars[32][1][1] --> the second action taken on trial 32
        #example: sars[9][2][0] --> reward at the end of the 9th trial
         
    for ep in range(len(sars)):
        ##rewarded: score is 10 
        if sars[ep][2][0] == 10 and ep<N -1 and ep>1:
            ##common --> transition leads to proper state
                #state 0 and action 0 leads to state 2
                #state 0 and action 1 leads to state 1
            if sars[ep][1][0] == 0 and sars[ep][0][1] == 2:
                if sars[ep][1][0] == sars[ep+1][1][0]:
                        common_rewarded.append(True)
                else:
                    common_rewarded.append(False)
            elif sars[ep][1][0] == 1 and sars[ep][0][1] == 1:
                if sars[ep][1][0] == sars[ep+1][1][0]:
                        common_rewarded.append(True)
                else:
                    common_rewarded.append(False)
            ##rare --> transition leads to rare next state
                #state 0 and action 0 leads to state 1
                #state 0 and action 1 leads to state 2
            elif sars[ep][1][0] == 0 and sars[ep][0][1] == 1:
                if sars[ep][1][0] == sars[ep+1][1][0]:
                       rare_rewarded.append(True)
                else:
                    rare_rewarded.append(False)
            elif sars[ep][1][0] == 1 and sars[ep][0][1] == 2:
                if sars[ep][1][0] == sars[ep+1][1][0]:
                        rare_rewarded.append(True)
                else:
                    rare_rewarded.append(False)
        ##unrewarded: score is 0
        if sars[ep][2][0] == 0 and ep<N-1 and ep>1:
            ##common
            if sars[ep][1][0] == 0 and sars[ep][0][1] == 2:
                if sars[ep][1][0] == sars[ep+1][1][0]:
                        common_unrewarded.append(True)
                else:
                    common_unrewarded.append(False)
            elif sars[ep][1][0] == 1 and sars[ep][0][1] == 1:
                if sars[ep][1][0] == sars[ep+1][1][0]:
                        common_unrewarded.append(True)
                else:
                    common_unrewarded.append(False)
            ##rare
            elif sars[ep][1][0] == 0 and sars[ep][0][1] == 1:
                if sars[ep][1][0] == sars[ep+1][1][0]:
                       rare_unrewarded.append(True)
                else:
                    rare_unrewarded.append(False)
            elif sars[ep][1][0] == 1 and sars[ep][0][1] == 2:
                if sars[ep][1][0] == sars[ep+1][1][0]:
                        rare_unrewarded.append(True)
                else:
                    rare_unrewarded.append(False)
    
    tot_cr = sum(common_rewarded)  
    crp = tot_cr/len(common_rewarded)

    tot_rr = sum(rare_rewarded)
    rrp = tot_rr/len(rare_rewarded)

    tot_cu = sum(common_unrewarded)
    cup = tot_cu/len(common_unrewarded)

    tot_ru = sum(rare_unrewarded)
    rup = tot_ru/len(rare_unrewarded)  
    total = [crp, rrp, cup, rup]
    r1 = 1,2,4,5
    x_pos = [1,2,4,5]
    plt.bar(r1, total, color = ['blue', 'red'])
    bars = ('com_rew', 'rare_rew', 'common_unrew', 'rare_unrew')
    plt.title("Q-learning")
    plt.xticks(x_pos, bars, rotation = 45, fontsize = 14)
    plt.ylim([.2,1])
    
    plt.show()   
    ##once again, the problem is that crp and rrp should always be higher than cup and rup
        #mine vary from run to run      

                




