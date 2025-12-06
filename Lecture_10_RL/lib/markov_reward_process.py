import random
import numpy as np
from .markov_chain import MarkovChain

# ------------------- State space + TransitionMatrix + Rewards + Discount factor ------------------- #
states  = ["sleep","wake up","eat","study","work out","watch tv","died","got a job"]
rewards = [     -1,       +1,    0,     +5,        +5,        -1,    -5,        +10]    
transitionMatrix = [
    [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.2, 0.0, 0.4, 0.2, 0.2, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.1, 0.4, 0.4, 0.1, 0.0, 0.0],
    [0.1, 0.0, 0.0, 0.4, 0.2, 0.1, 0.0, 0.2],
    [0.2, 0.0, 0.2, 0.0, 0.5, 0.0, 0.1, 0.0],
    [0.2, 0.0, 0.0, 0.0, 0.0, 0.3, 0.5, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]
# gamma is the discount factor
# and is a hyperparameter of a Markov Reward Process
# it is used to discount the future rewards
# when gamma is 0 the agent only cares about the immediate reward
# when gamma is 1 the agent cares about all the future rewards
# when gamma is between 0 and 1 the agent cares about the immediate reward and the future rewards
gamma = 0.2

class MarkovRewarProcess(MarkovChain):
    def __init__(self, states: list[str], transitionMatrix: list[list[float]], rewards: list[int],gamma: float):
        super().__init__(states, transitionMatrix)
        self.R = rewards
        self.gamma = gamma

    def verify(self):
        super().verify()
        if self.gamma <= 0 or self.gamma > 1:
            raise ValueError("Gamma must be in range [0,1)")
        if len(self.R) != len(self.S):
            raise ValueError("The number of states and rewards must be the same")

    def sampleCumlativeReward(self,episode,log=False):
        G_t = 0
        for i in range(len(episode)-1):
            state = self.S.index(episode[i])
            G_t += self.R[state] * (self.gamma ** i)
            if log:
                print("G_t = {:.4f}, gamma^k = {:.4f}".format( G_t, gamma**i ))
        return G_t

    def solve_MRPvalueFunction_by_iteration(self,log=False):
        V = np.zeros(len(self.S))
        num_episodes = 2000
        # solving the value function by iteration
        # is the same as solving the Bellman equation
        # V(s) = R(s) + gamma * sum(T(s,s') * V(s'))
        # where 
        # R(s) is the reward of state s
        # s' is the next state
        # T(s,s') is the transition probability from s to s' aka the transition matrix
        # V(s') is the value of the next state
        # gamma is the discount factor
        # NOTE: "sum" is a discounted sum of the future rewards
        # which means that the value of the future rewards decreases as we go further in the future
        # this is because the value of the future rewards is multiplied by gamma (0 <= gamma < 1)
        for k in range(num_episodes):
            for i in range(len(self.S)):
                state = self.S[i]
                # each time we sample an episode we get a different sequence of states
                # that the agent takes from the state being evaluated to the terminal state
                # based upon the transition matrix's probabilities
                episode = self.sample_episode(state,log=False)
                # after sampling an episode we calculate the cumulative reward for that episode
                V[i] += self.sampleCumlativeReward(episode,log=False)

            if log and (k + 1) % 100 == 0:
                np.set_printoptions(precision=2)
                print(V/(k+1))
        V = V/num_episodes
        print(V)
    
    def solve_MRPvalueFunction_by_BellmanEquation(self):
        # solving the value function by the Bellman equation:
        # the Belleman can be expressed with matrices
        # V = R + gamma * T * V
        # V - gamma * T * V = R
        # (I - gamma * T) * V = R 
        # V = (I - gamma * T)^-1 * R
        I = np.identity(len(self.T))
        V = np.linalg.solve(I - self.gamma * np.array(self.T), np.array(self.R))
        # however this method is not recommended for large state spaces
        # because the complexity of the matrix inversion is O(n^3)
        # where n is the number of states
        print(V)

    def reward(self):
        index = self.S.index(self.currentState)
        return self.R[index]


# ------------------- Main ------------------- #
if __name__ == "__main__":
    markovRewardProcess = MarkovRewarProcess(states,transitionMatrix,rewards,gamma)
    markovRewardProcess.verify()
    # the computional complexity of the iteration method is O(n^2)
    # for each iteration (episode)
    # where 
    # n is the number of states
    markovRewardProcess.solve_MRPvalueFunction_by_iteration(log=False)

    # the computational complexity of the Bellman equation is O(n^3)
    # where
    # n is the number of states
    markovRewardProcess.solve_MRPvalueFunction_by_BellmanEquation()