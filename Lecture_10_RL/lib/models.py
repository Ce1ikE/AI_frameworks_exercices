from enum import Enum

import torch
import torch.nn as nn

class AlgorithmType(Enum):
    DQN = "DQN"
    Q_LEARNING = "Q_learning"
    MCMC = "Markov_Chain_Monte_Carlo"
    VALUE_ITERATION = "Value_Iteration"
    POLICY_ITERATION = "Policy_Iteration"
    # TODO: 
    ACTOR_CRITIC = "Actor_Critic"
    MC_POLICY_EVALUATION = "Monte_Carlo_Policy_Evaluation"
    INCREMENTAL_MC_POLICY_EVALUATION = "Incremental_Monte_Carlo_Policy_Evaluation"

class Algorithm:
    def __init__(self, algorithm_type: AlgorithmType):
        self.algorithm_type = algorithm_type

    def train(self):
        raise NotImplementedError("Train method must be implemented by subclasses.")

    def evaluate(self):
        raise NotImplementedError("Evaluate method must be implemented by subclasses.")

    def settings(self) -> dict:
        raise NotImplementedError("Settings method must be implemented by subclasses.")

class DeepQNetwork(nn.Module):
    def __init__(self, n_states, m_actions, hidden_dim=64):
        super(DeepQNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Embedding(n_states, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, m_actions)
        )

    def forward(self, x) -> torch.Tensor:
        return self.net(x)
    

