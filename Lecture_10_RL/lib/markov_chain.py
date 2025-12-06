import random

# ------------------- State space + TransitionMatrix ------------------- #
states = [
    "sleep","wake up","eat","study","work out","watch tv","died","got a job"
]
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

# ------------------- Markov Chain ------------------- #
class MarkovChain:
    def __init__(self,states: list[str],transitionMatrix: list[list[float]]):
        self.S = states
        self.T = transitionMatrix
        
    def verify(self):
        if len(self.S) != len(self.T):
            raise ValueError("The number of states and transition probabilities must be the same")
        for i in range(len(self.T)):
            if len(self.S) != len(self.T[i]):
                raise ValueError("The number of states and transition probabilities must be the same")
            if sum(self.T[i]) != 1:
                raise ValueError("The sum of transition probabilities for state {} must be 1".format(self.S[i]))

    def next_state(self):
        index = self.S.index(self.currentState)
        self.currentState = random.choices(self.S,self.T[index])[0]
        return self.currentState
    
    def sample_episode(self,state="sleep",log=False):
        self.currentState = state
        episode = [state]
        while episode[-1] != "died" and episode[-1] != "got a job":
            next_state = self.next_state()
            episode.append(next_state)
        if log:
            print(episode)
        return episode
    

# ------------------- Main ------------------- #
if __name__ == "__main__":
    markovChain = MarkovChain(states,transitionMatrix)
    markovChain.verify()
    number_of_episodes = 5
    for i in range(number_of_episodes):
        markovChain.sample_episode(log=True)

