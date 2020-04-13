import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import gym 
 

 
import hiive.mdptoolbox.mdp as mdp 
import mdptoolbox.example


from time import time
import pickle

if __name__ == "__main__": 
 
    run()


def run():
     
    envNames = ['FrozenLake-v0', 'FrozenLake8x8-v0', 'Forest Management'] 
    for envName in envNames:
        maxIters = 1100000
        maxIters = 600000
        maxIters = 13000
        maxIters = 10003
        itRange = range(10000, maxIters, 250000)
        itRange = range(10000, maxIters, 250000)
        itRange = np.linspace(50000, maxIters, 3)
        # itRange = range(10000, 1000000, 100000)
        itRange = np.linspace(10000, maxIters, 3)
        # itRange = [10000,20000,30000,40000,50000]
        # itRange = [10000,250000,500000]
        # itRange = np.linspace(10000, 700000, 10)
        if envName == 'Forest Management':
            discountRange = np.linspace(.01, .99, 99)
            transitions, rewards = mdptoolbox.example.forest(S=1000) 
        else:
            discountRange = np.linspace(.01, .99, 99) 
            env = gym.make(envName, is_slippery=False) 
            states = env.observation_space.n
            actions = env.action_space.n
            transitions, rewards = createTransitionsAndRewards(env, states, actions)  
 

        print('{0} Value iteration'.format(envName)) 
        valuePolicy = runValue(transitions, rewards, envName, maxIters, discountRange)
        print('{0} Policy iteration'.format(envName)) 
        policyPolicy = runPolicy(transitions, rewards, envName, maxIters, discountRange)
 
 
        itRange = np.linspace(10000, 700000, 10)
        itRange = [int(x) for x in itRange] 
        print('{0} Epsilon by iterations'.format(envName)) 
        epsPolicy = runQEpsilonByIts(transitions, rewards, envName, itRange)
        print('{0} Epsilon by alphas'.format(envName)) 
        alphaPolicy = runQAlphaByIts(transitions, rewards, envName, itRange) 
        print('{0} Iterations by discount'.format(envName)) 
        itRange = np.linspace(10000, 700000, 4)
        itRange = [int(x) for x in itRange]
        discountPolicy = runQItsByDiscount(transitions, rewards, envName, itRange)
        
        


        valueStats = scorePolicy(valuePolicy, env)
        policyStats = scorePolicy(policyPolicy, env)
        # qStats = playPolicy(qPolicy, env)
        print('{0} done'.format(envName))
    
    print('done')
   
def printStat(stat):
    # stat = [i, discount, alg.time, runStats['Iteration'], runStats['Error'], np.mean(alg.V), runStats['Reward']]
    # print('Discount: {0} \t\t Reward: {1} \t Wins: {2} \t Time: {3}'.format(stat[1], stat[5], stat[6], stat[2])) 
    # print('Discount:{0:10.4f} \t Time:{1:10.4f} \t Iterations:{2:10.0f} \t Error:{3:10.4f} \t Reward:{4:10.4f} \t Wins:{5:10.4f}'
    # .format(stat[1],stat[2],stat[3], stat[4], stat[5], stat[6]))     
    print('Discount:{0:10.4f} \t Time:{1:10.4f} \t Iterations:{2:10.0f} \t Error:{3:10.4f} \t Reward:{4:10.4f}'.format(stat[1],stat[2],stat[3], stat[4], stat[5])) 

def save(title, data):
    bbb = 34
    # basePath = ''
    # filePath = basePath + title + '.pickle'  
 
    # with open(filePath, 'wb') as outfile:  
    #     pickle.dump(data, outfile)
    #     outfile.close()


def show(title, fig, ax):
    basePath = ''
    # plt.show()

  ############  # if fig: 

      ######  # handles, labels = ax.get_legend_handles_labels()
      ###########  # lgd = ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 0.5)) 
      ###########  # fig.savefig('{0}{1}.png'.format(basePath, title), bbox_extra_artists=('',lgd), bbox_inches='tight')

    plt.xticks(rotation=45)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 0.5), bbox_transform=ax.transAxes)
    plt.tight_layout()
    ################### plt.legend(loc="best")
    # plt.savefig('{0}{1}.png'.format(basePath, title)) 
    plt.show()
    plt.clf()
    plt.close()
       ############ # fig.savefig('{0}{1}.png'.format(basePath, title))
  #########  # else:
   ########## #     plt.savefig('{0}{1}.png'.format(basePath, title)) 
   ########## #     plt.clf()
         
 
def plotQAx(ax, x, y, title, xLabel, yLabel, seriesLabel): 
    
    ax.plot(x, y, label=seriesLabel) 

    # for i in series:
    #     ax.plot(x, y, label='{0} {1}'.format(i, seriesLabel)) 
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_title(title)
    ax.grid() 
    
    return ax

             
 
def plotAx(x, y, title, xLabel, yLabel): 
    fig, ax = plt.subplots() 
    ax.plot(x, y) 
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_title(title)
    ax.grid() 
    return fig, ax
    # show(title, fig, ax) 
  


def createTransitionsAndRewards(environment, states, actions):


    # S=n_states, r1=4, r2=2, p=0.1, is_sparse=False):
    # P = _np.zeros((2, S, S))
    # P[0, :, :] = (1 - p) * _np.diag(_np.ones(S - 1), 1)
    # P[0, :, 0] = p
    # P[0, S - 1, S - 1] = (1 - p)
    # P[1, :, :] = _np.zeros((S, S))
    # P[1, :, 0] = 1
    # # Definition of Reward matrix
    # R = _np.zeros((S, 2))
    # R[S - 1, 0] = r1
    # R[:, 1] = _np.ones(S)
    # R[0, 1] = 0
    # R[S - 1, 1] = r2
    # return(P, R)
    # print(environment.env.P)
 
    rewards = np.zeros((states, actions))
    probabilities = np.zeros((actions, states, states))
 
    for s in range(states):
        for a in range(actions):
            for data in environment.env.P[s][a]:
                prob, sprime, reward, done = data
                if done and reward == 1:
                    reward = 10000
                    # reward = 1
                elif done and reward == 0:
                    reward = -1000
                    # reward = 0
                else:
                    reward = -10
                rewards[s, a] = reward
                probabilities[a, s, sprime] = prob

                prob = probability_matrix[a, s, :]
                probSum = np.sum(probability_matrix[a, s, :])
                probabilities[a, s, :] = prob / probSum 

    return probabilities, rewards




def runValue(transitions, rewards, envName, maxIters, discountRange):
    stats = []
    for discount in discountRange:
        alg = mdp.ValueIteration(transitions, rewards, discount, max_iter=maxIters)
        result = alg.run()
        runStats = alg.run_stats[-1]
        stats.append([discount, alg.time, runStats['Iteration'], runStats['Error'], runStats['Reward'],  np.mean(alg.V), alg.policy])
        # stats.append([discount, alg.time, runStats['Iteration'], alg.error_mean, runStats['Reward'], alg.v_mean, alg.policy])

    statsT = list(zip(*stats))
    discounts = statsT[0]
    times = statsT[1]
    iterations = statsT[2]
    errors = statsT[3]
    wins = statsT[4]
    rewards = statsT[5]
 
    stats = sorted(stats, key= lambda x: x[5], reverse=True)
    topPolicy = stats[0]
    topPolicy = topPolicy[-1] 
 
    title = '{0} Value Iteration - Time'.format(envName)
    fig, ax = plotAx(discounts, times, title, 'Discount Factor', 'Time')
    show(title, fig, ax)
 

    title = '{0} Value Iteration - Iterations'.format(envName)
    fig, ax = plotAx(discounts, iterations, title, 'Discount Factor', 'Iteration')
    show(title, fig, ax)
 
    title = '{0} Value Iteration - Error'.format(envName)
    fig, ax = plotAx(discounts, errors, title, 'Discount Factor', 'Error')
    show(title, fig, ax)
    
    title = '{0} Value Iteration - Reward'.format(envName)
    fig, ax = plotAx(discounts, rewards, title, 'Discount Factor', 'Rewards')
    show(title, fig, ax)
 
    return topPolicy


def runPolicy(transitions, rewards, envName, maxIters, discountRange):
    stats = []
    for discount in discountRange: 
        alg = mdp.PolicyIteration(transitions, rewards, discount, max_iter=maxIters)
        result = alg.run() 
        runStats = alg.run_stats[-1]
        stats.append([discount, alg.time, runStats['Iteration'], runStats['Error'], runStats['Reward'],  np.mean(alg.V), alg.policy])

    statsT = list(zip(*stats))
    discounts = statsT[0]
    times = statsT[1]
    iterations = statsT[2]
    errors = statsT[3]
    wins = statsT[4]
    rewards = statsT[5]
 

    stats = sorted(stats, key= lambda x: x[5], reverse=True)
    topPolicy = stats[0]
    topPolicy = topPolicy[-1] 

 
    title = '{0} Policy Iteration - Time'.format(envName)
    fig, ax = plotAx(discounts, times, title, 'Discount Factor', 'Time')
    show(title, fig, ax)
 

    title = '{0} Policy Iteration - Iterations'.format(envName)
    fig, ax = plotAx(discounts, iterations, title, 'Discount Factor', 'Iteration')
    show(title, fig, ax)
 
    title = '{0} Policy Iteration - Error'.format(envName)
    fig, ax = plotAx(discounts, errors, title, 'Discount Factor', 'Error')
    show(title, fig, ax)
    
    title = '{0} Policy Iteration - Reward'.format(envName)
    fig, ax = plotAx(discounts, rewards, title, 'Discount Factor', 'Rewards')
    show(title, fig, ax)
    return topPolicy
 
  

def runQItsByDiscount(transitions, rewards, envName, itRange): 

    discountRange = np.linspace(.01, .99, 20) 
    # discountRange = [.01, .1, .25, .5, .90, .98]
 
    stats = []
    for i in itRange:  
        print()
        print('{0} Itt: {1}'.format(envName,i))
        for discount in discountRange:  
            alg = mdp.QLearning(transitions, rewards, discount, n_iter=i)
            result = alg.run() 
            runStats = alg.run_stats[-1]
            stat = [i, discount, alg.time, runStats['Iteration'], runStats['Error'], np.mean(alg.V), runStats['Reward'], alg.policy]
            printStat(stat)
            stats.append(stat) 
            save('{0} {1} discount policy'.format(envName, i), alg.policy)

    statsArr = np.array(stats)
    save('{0} discount stats'.format(envName), statsArr)
  
    roundedDiscounts = [round(x, 3) for x in discountRange]
    title = '{0} Q Learning - Time by Discount Factor'.format(envName)
    fig, ax = plt.subplots() 
    for i in itRange: 
        iStats = statsArr[statsArr[:, 0] == i]   
        times = iStats[:, 2]   
        plotQAx(ax, discountRange, times, title,  'Discount Factor', 'Time', 'iterations {0}'.format(i))
    show(title, fig, ax)
   
    title = '{0} Q Learning - Error by Discount Factor'.format(envName)  
    fig, ax = plt.subplots() 
    for i in itRange: 
        iStats = statsArr[statsArr[:, 0] == i]  
        errors = iStats[:, 4]    
        plotQAx(ax, discountRange, errors, title,  'Discount Factor', 'Error', 'iterations {0}'.format(i))
    show(title, fig, ax)
         
    title = '{0} Q Learning - Reward by Discount Factor'.format(envName)  
    fig, ax = plt.subplots() 
    for i in itRange: 
        iStats = statsArr[statsArr[:, 0] == i]   
        rewards = iStats[:, 5]  
        plotQAx(ax, discountRange, errors, title,  'Discount Factor', 'Reward', 'iterations {0}'.format(i))  
    show(title, fig, ax)         
 
  
    stats = sorted(stats, key= lambda x: x[5], reverse=True)
    topPolicy = stats[0]
    topPolicy = topPolicy[-1] 
    return topPolicy

def runQEpsilonByIts(transitions, rewards, envName, itRange):
    epsilons = np.linspace(.9, .01, 10)
    epsilons =  [.01, .1, .25, .5, .90, .98] 
    discounts = [.1, .5, .9]
    allStats = []
    for discount in discounts:
        stats = []
        for i in epsilons: 
            print() 
            print('{0} Epsilon: {1} Discount:{2}'.format(envName,i, discount))
            for itt in itRange:  
                alg = mdp.QLearning(transitions, rewards, discount, epsilon=i, n_iter=itt)
                result = alg.run() 
                runStats = alg.run_stats[-1]
                stat = [i, discount, alg.time, runStats['Iteration'], runStats['Error'], np.mean(alg.V), runStats['Reward']]
                printStat(stat)
                stats.append(stat)   
                save('{0} {1} epsilon {2} discount policy'.format(envName, i, discount), alg.policy)
        statsArr = np.array(stats) 
        save('{0} epsilon {1} discount stats'.format(envName, discount), statsArr)
    
        
        roundedEpsilons = [round(x, 3) for x in epsilons]
        title = '{0} Q Learning - Time by Epsilon Discount {1}'.format(envName, discount)
        fig, ax = plt.subplots() 
        for i in epsilons: 
            iStats = statsArr[statsArr[:, 0] == i]   
            times = iStats[:, 2]   
            plotQAx(ax, itRange, times,  title,  'Iterations', 'Time', 'epsilon {0:0.3f}'.format(i))
        show(title, fig, ax)

        title = '{0} Q Learning - Error by Epsilon Discount {1}'.format(envName, discount)  
        fig, ax = plt.subplots() 
        for i in epsilons: 
            iStats = statsArr[statsArr[:, 0] == i]  
            errors = iStats[:, 4]    
            plotQAx(ax, itRange, errors,  title,  'Iterations', 'Error', 'epsilon {0:0.3f}'.format(i))
        show(title, fig, ax)
            
        title = '{0} Q Learning - Reward by Epsilon Discount {1}'.format(envName, discount)  
        fig, ax = plt.subplots() 
        for i in epsilons: 
            iStats = statsArr[statsArr[:, 0] == i]   
            rewardsArr = iStats[:, 5]  
            plotQAx(ax, itRange, rewardsArr, title,  'Iterations', 'Reward', 'epsilon {0:0.3f}'.format(i))
        show(title, fig, ax)    
        allStats.extend(stats)

    allStats = sorted(allStats, key= lambda x: x[5], reverse=True)
    topPolicy = stats[0]
    topPolicy = topPolicy[-1] 
    return topPolicy
   
def runQAlphaByIts(transitions, rewards, envName, itRange): 
    alphas = np.linspace(.9, .01, 10)
    alphas =  [.01, .05, .10, .20, .25]  
    discounts = [.1, .5, .9]
    allStats = []
    for discount in discounts:
        stats = []
        for i in alphas:  
            print()
            print('{0} Alpha: {1} Discount:{2}'.format(envName,i, discount))
            for itt in itRange:  
                alg = mdp.QLearning(transitions, rewards, discount, alpha=i, n_iter=itt)
                result = alg.run() 
                runStats = alg.run_stats[-1]
                stat = [i, discount, alg.time, runStats['Iteration'], runStats['Error'], np.mean(alg.V), runStats['Reward']]
                printStat(stat)
                stats.append(stat)   
                save('{0} {1} epsilon discount{2} policy'.format(envName, i, discount), alg.policy)
        statsArr = np.array(stats) 
        save('{0} {1} alpha stats'.format(envName, discount), statsArr)
    
        
        roundedAlphas = [round(x, 3) for x in alphas]
        title = '{0} Q Learning - Time by Alpha Discount {1}'.format(envName, discount)
        fig, ax = plt.subplots() 
        for i in alphas: 
            iStats = statsArr[statsArr[:, 0] == i]   
            times = iStats[:, 2]   
            plotQAx(ax, itRange, times, title,  'Iterations', 'Time', 'alpha {0:0.3f}'.format(i))
        show(title, fig, ax)
    

        title = '{0} Q Learning - Error by Alpha Discount {1}'.format(envName, discount)  
        fig, ax = plt.subplots() 
        for i in alphas: 
            iStats = statsArr[statsArr[:, 0] == i]  
            errors = iStats[:, 4]    
            plotQAx(ax, itRange, errors, title,  'Iterations', 'Error', 'alpha {0:0.3f}'.format(i))
        show(title, fig, ax)
            
        title = '{0} Q Learning - Reward by Alpha Discount {1}'.format(envName, discount)  
        fig, ax = plt.subplots() 
        for i in alphas: 
            iStats = statsArr[statsArr[:, 0] == i]   
            rewardsArr = iStats[:, 5]  
            plotQAx(ax, itRange, rewardsArr, title,  'Iterations', 'Reward', 'alpha {0:0.3f}'.format(i))  
        show(title, fig, ax)   
        allStats.extend(stats)

    allStats = sorted(allStats, key= lambda x: x[5], reverse=True)
    topPolicy = stats[0]
    topPolicy = topPolicy[-1] 
    return topPolicy
          
def scorePolicy(env, policy, render=False):
    done = False
    wins = 0
    total = 0

    state = env.reset()
    while not done:
        action = policy[state]
        sPrime, reward, done, information = env.step(action)
        if render:
            env.render()
        total += reward
        if done and reward == 1:
            wins += 1
        
    return total, wins

