from time import time
from typing import Any
from collections import namedtuple, deque
from tqdm import tqdm

import time
import torch
import gymnasium as gym
import numpy as np

from .global_const import *
from .models import *
from .utils import Utils

class DQNAlgorithm(Algorithm):
    """
    Deep Q-Network Algorithm using Google DeepMind's DQN approach.
    Which uses experience replay buffer and a target network to stabilize training.
    https://github.com/google-deepmind/dqn_zoo
    """

    def __init__(
        self, 
        num_episodes: int = 1000,
        max_steps_per_episode: int = 100,
        learning_rate: float = 0.001,
        discount_rate: float = 0.99,
        exploration_rate: float = 1.0,
        max_exploration_rate: float = 1.0,
        min_exploration_rate: float = 0.01,
        exploration_decay_rate: float = 0.001,
        replay_buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_frequency: int = 10,
        warmup_steps: int = 100,
    ):
        """
        Args:
            num_episodes (int): Number of episodes to train the DQN.
            max_steps_per_episode (int): Maximum steps per episode.
            learning_rate (float): Learning rate for the optimizer.
            discount_rate (float): Discount factor for future rewards.
            exploration_rate (float): Initial exploration rate for epsilon-greedy policy.
            max_exploration_rate (float): Maximum exploration rate.
            min_exploration_rate (float): Minimum exploration rate.
            exploration_decay_rate (float): Decay rate for exploration probability.
            replay_buffer_size (int): Size of the experience replay buffer.
            batch_size (int): Batch size for training.
            target_update_frequency (int): Frequency to update the target network.
            warmup_steps (int): Number of steps to populate the replay buffer before training.
        """

        super(DQNAlgorithm, self).__init__(AlgorithmType.DQN)
        # input dimension = observation space dimension but one-hot encoded
        # output dimension = action space dimension
        self.model = None
        # the target model is used to stabilize training
        # we set it to evaluation mode to disable certain layers like dropout and batchnorm
        self.target_model = None
        self.optimizer = None
        self.loss_fn = None
        # we also need a replay buffer to store the experiences
        # and sample mini-batches from it during training
        self.replay_buffer_size = replay_buffer_size
        # size (buffer_size, 5) for (state, action, reward, new_state, done)
        # we use a numpy array for efficiency instead of a list or deque
        self.experience_replay_buffer = np.zeros((replay_buffer_size,5), dtype=np.float32)
        self.ptr = 0
        self.full = False
        
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.exploration_rate = exploration_rate
        self.max_exploration_rate = max_exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.warmup_steps = warmup_steps
        self.device = None
        
        # evaluation metrics
        self.rewards_all_episodes = []
        self.steps_per_episode = []

    def train(
        self,
        env: gym.Env
    ):
        """
        Args:
            env (gym.Env): The environment to train the DQN on. (e.g., FrozenLakeEnv)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # input dimension = observation space dimension but one-hot encoded
        # output dimension = action space dimension
        self.model = DeepQNetwork(env.observation_space.n, env.action_space.n).to(self.device)
        # the target model is used to stabilize training
        # we set it to evaluation mode to disable certain layers like dropout and batchnorm
        self.target_model = DeepQNetwork(env.observation_space.n, env.action_space.n).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = torch.nn.MSELoss().to(self.device)

        # the most difficult part about integrating a NN in RL is actually 
        # correctly placing the training step and the experience replay buffer 
        # since we always have our main loop that interacts with the environment
        # however the training step is not done in every algorithm like in Q-learning

        # basically we just store the experiences in a replay buffer that will act as a memory for sampling batches of experiences later for training
        # then we implement a training step (pytorch training step) 
        # that samples a batch of experiences from the replay buffer and performs a gradient descent step to update the neural network weights

        # what we do add is a target network that is a copy of the main network
        # that is used to compute the target Q-values during training
        # this target network is updated periodically with the weights of the main network
        # this helps to stabilize the training since the target Q-values are not changing too rapidly

        # DQN algorithm steps:
        # for each episode:
        #    for each step in the episode:
        #        choose action (a) using a policy derived from Q (e.g., epsilon-greedy) 
        #        take action (a), observe reward (r) and new state (s')
        #        store experience (s, a, r, s', done) in replay buffer
        #        sample random mini-batch of experiences from replay buffer
        #        (main training step like in normal pytorch training loop) 
        #        perform a gradient descent step on the loss between predicted Q-values and target Q-values
        #    update target network periodically
        gen = np.random.default_rng(RANDOM_SEED)
        for episode in tqdm(range(self.num_episodes), desc="Training DQN", unit="episode"):
            state, info = env.reset()
            done = False
            rewards_current_episode = 0

            for step in range(self.max_steps_per_episode):
                
                # Epsilon greedy algorithm
                if gen.uniform(0,1) > self.exploration_rate:
                    state_tensor = torch.LongTensor([state]).to(self.device)
                    with torch.no_grad():
                        q_values = self.model(state_tensor)
                        action = torch.argmax(q_values).item()
                else:
                    action = env.action_space.sample()
                new_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # store experience in replay buffer
                # experience = Experience(state, action, reward, new_state, done)
                self.experience_replay_buffer[self.ptr] = (state, action, reward, new_state, float(done))
                self.ptr = (self.ptr + 1) % self.replay_buffer_size

                if self.ptr == 0:
                    self.full = True

                buffer_size = self.replay_buffer_size if self.full else self.ptr

                # training step (very similar to normal pytorch training loop)
                if buffer_size >= max(self.batch_size, self.warmup_steps):
                    # first we sample a mini-batch of experiences from the replay buffer
                    idx = torch.randint(0, buffer_size, (self.batch_size,))
                    # boiler plate to convert batch data to tensors (this part can probably be optimized)
                    batch = self.experience_replay_buffer[idx]
                    states = torch.LongTensor(batch[:, 0]).to(self.device)
                    actions = torch.LongTensor(batch[:, 1]).unsqueeze(1).to(self.device)
                    rewards = torch.FloatTensor(batch[:, 2]).unsqueeze(1).to(self.device)
                    next_states = torch.LongTensor(batch[:, 3]).to(self.device)
                    dones = torch.FloatTensor(batch[:, 4]).unsqueeze(1).to(self.device)
                    # then we compute the current Q-values and the target Q-values
                    # using the same bellman equation as in Q-learning
                    # if the episode is done, the target Q-value is just the reward
                    current_q_values = self.model(states).gather(1, actions)
                    # no_grad to avoid backprop through the target network which is not being trained
                    with torch.no_grad():
                        next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
                    target_q_values = rewards + (self.discount_rate * next_q_values * (1 - dones))
                    # compute the loss between current Q-values and target Q-values
                    loss = self.loss_fn(current_q_values, target_q_values)
                    # perform a gradient descent step to update the model weights
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                state = new_state
                rewards_current_episode += reward

                if done:
                    break

            # and finally we update the exploration rate and the target network periodically
            if episode % self.target_update_frequency == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            # Exploration rate decay update
            self.exploration_rate = self.min_exploration_rate + (self.max_exploration_rate - self.min_exploration_rate) * np.exp(-self.exploration_decay_rate * episode)
            self.rewards_all_episodes.append(rewards_current_episode)
                    
        Utils.plot_avg_rewards(
            self.rewards_all_episodes, 
            window_size=100, 
            algorithm=self.algorithm_type
        )
        Utils.save_model(
            self.model, 
            example_input=torch.LongTensor([0]).to(self.device),
            filename="dqn_model.pth", 
            algorithm=self.algorithm_type
        )

    def evaluate(
        self,
        env: gym.Env,
        num_episodes: int = 100,
        max_steps_per_episode: int = 100,
        time_sleep_between_episodes: float = 0.5,
        time_sleep_between_steps: float = 0.1,
        goal_reward: float = 1.0,
    ):
        if self.model is None:
            raise ValueError("The DQN model is not initialized. Please train the agent before evaluation.")
        
        self.model.eval()
        rewards_all_episodes = []
        successful_episodes = 0
        steps_taken = 0

        for episode in range(num_episodes):
            state, info = env.reset()
            done = False
            print(f"*****EPISODE {episode + 1} *****")
            time.sleep(time_sleep_between_episodes)

            rewards_current_episode = 0
            for step in range(max_steps_per_episode):
                time.sleep(time_sleep_between_steps)
        
                state_tensor = torch.LongTensor([state]).to(self.device)
                with torch.no_grad():
                    q_values = self.model(state_tensor)
                    action = torch.argmax(q_values).item()
                new_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                rewards_current_episode += reward
                steps_taken += 1

                if done:
                    if reward == goal_reward:
                        print("****\tYou reached the goal!\t****")
                        time.sleep(time_sleep_between_episodes)
                        successful_episodes += 1
                    else:
                        print("****\tYou failed!\t****")
                        time.sleep(time_sleep_between_episodes)
                    break

                if step == max_steps_per_episode - 1:
                    print("****\tMaximum steps reached!\t****")
                    time.sleep(time_sleep_between_episodes)

                state = new_state
            rewards_all_episodes.append(rewards_current_episode)
        print(f"Evaluated over {num_episodes} episodes, average reward: {np.mean(rewards_all_episodes):.2f}, average steps per episode: {steps_taken / num_episodes:.2f}")

        return {
            "mean_reward": np.mean(rewards_all_episodes),
            "successful_episodes": successful_episodes,
            "num_episodes": num_episodes,
            "success_rate": successful_episodes / num_episodes,
            "max_steps_per_episode": max_steps_per_episode,
            "average_steps_per_episode": steps_taken / num_episodes,
        }

    def settings(self) -> dict:
        return {
            "num_episodes": self.num_episodes,
            "max_steps_per_episode": self.max_steps_per_episode,
            "learning_rate": self.learning_rate,
            "discount_rate": self.discount_rate,
            "exploration_rate": self.exploration_rate,
            "max_exploration_rate": self.max_exploration_rate,
            "min_exploration_rate": self.min_exploration_rate,
            "exploration_decay_rate": self.exploration_decay_rate,
            "replay_buffer_size": self.replay_buffer_size,
            "batch_size": self.batch_size,
            "target_update_frequency": self.target_update_frequency,
            "warmup_steps": self.warmup_steps,
        }

class QLearningAlgorithm(Algorithm):
    """
    Q-Learning Algorithm (off-policy) 
    using epsilon-greedy policy with decaying exploration rate.
    """
    def __init__(
        self,
        num_episodes: int = 1000,
        max_steps_per_episode: int = 100,
        learning_rate: float = 0.001,
        discount_rate: float = 0.99,
        exploration_rate: float = 1.0, 
        max_exploration_rate: float = 1.0, 
        min_exploration_rate: float = 0.01,
        exploration_decay_rate: float = 0.001,
    ):
        super(QLearningAlgorithm, self).__init__(AlgorithmType.Q_LEARNING)
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.exploration_rate = exploration_rate
        self.max_exploration_rate = max_exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.q_table = None
        self.history = []
        # evaluation metrics
        self.rewards_all_episodes = []
        self.q_values_history = []

    def train(
        self,
        env: gym.Env  
    ):
        gen = np.random.default_rng(RANDOM_SEED)
        # Q-learning algorithm (off-policy)
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        # Q Table structure:
        #     |   a0   |   a1   |  ...  |   a_n  | 
        #     |--------|--------|       |--------|
        # s0  |   0    |   0    |  ...  |   0    | 
        # s1  |   0    |   0    |  ...  |   0    | 
        # ..  |  ...   |  ...   |  ...  |  ...   |
        # s_m |   0    |   0    |  ...  |   0    |

        # Q-learning algorithm steps:
        # for each episode:
        #  for each step in the episode:
        #  choose action (a) using epsilon-greedy policy from Q-table
        #  take action (a), observe reward (r) and new state (s')
        #  update Q-table using the Q-learning formula
        #  update state to new state (s = s') 
        for episode in range(self.num_episodes):
            state, info = env.reset()
            done=False
            rewards_current_episode = 0
            self.history.append(self.q_table.copy())
            for step in range(self.max_steps_per_episode):

                # Epsilon greedy algorithm
                if gen.uniform(0,1) > self.exploration_rate:
                    action = np.argmax(self.q_table[state,:])
                else:
                    action = env.action_space.sample()

                new_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                # update Q-table for Q(s,a)
                # <=> Q(S,A) = (1 - alpha) * Q(S,A) + alpha * [R + gamma * max Q(S',a)] 
                # <=> Q(S,A) = Q(S,A) - alpha * Q(S,A)  + alpha * [R + gamma * max Q(S',a)] 
                # <=> Q(S,A) = Q(S,A) + alpha * [R + gamma * max Q(S',a) - Q(S,A)]
                self.q_table[state, action] = self.q_table[state, action] + self.learning_rate * (reward + self.discount_rate * np.max(self.q_table[new_state, :]) - self.q_table[state, action])

                # move to the new state 
                state = new_state
                rewards_current_episode += reward
                if done == True:
                    break

            # Exploration rate decay update
            self.exploration_rate = self.min_exploration_rate + (self.max_exploration_rate - self.min_exploration_rate) * np.exp(-self.exploration_decay_rate * episode)
            self.rewards_all_episodes.append(rewards_current_episode)  

        Utils.plot_q_value_convergence(
            np.array(self.history), 
            algorithm=self.algorithm_type
        )
        Utils.plot_avg_rewards(
            self.rewards_all_episodes, 
            window_size=100, 
            algorithm=self.algorithm_type
        )

    def evaluate(
        self,
        env: gym.Env,
        num_episodes: int,
        max_steps_per_episode: int,
        time_sleep_between_episodes: float = 0.5,
        time_sleep_between_steps: float = 0.1,
        goal_reward: float = 1.0,
    ):
        """
        Args:
            env (gym.Env): The environment to evaluate the Q-learning agent on. (e.g., FrozenLakeEnv)
            num_episodes (int): Number of episodes to run for evaluation.   
            max_steps_per_episode (int): Maximum steps allowed per episode.
            time_sleep_between_episodes (float): Time to sleep between episodes for better visualization.
            time_sleep_between_steps (float): Time to sleep between steps for better visualization.
        """

        if self.q_table is None:
            raise ValueError("The Q-table is not initialized. Please train the agent before evaluation.")

        rewards_all_episodes = []
        successful_episodes = 0
        steps_taken = 0

        for episode in range(num_episodes):
            state, info = env.reset()
            done = False
            print(f"*****EPISODE {episode + 1} *****")
            time.sleep(time_sleep_between_episodes)

            rewards_current_episode = 0
            for step in range(max_steps_per_episode):
                time.sleep(time_sleep_between_steps)
        
                action = np.argmax(self.q_table[state, :])
                new_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                rewards_current_episode += reward
                steps_taken += 1
                if done:
                    if reward == goal_reward:
                        print("****\tYou reached the goal!\t****")
                        time.sleep(time_sleep_between_episodes)
                        successful_episodes += 1
                    else:
                        print("****\tYou failed!\t****")
                        time.sleep(time_sleep_between_episodes)
                    break

                if step == max_steps_per_episode - 1:
                    print("****\tMaximum steps reached!\t****")
                    time.sleep(time_sleep_between_episodes)

                state = new_state
            rewards_all_episodes.append(rewards_current_episode)

        print(f"Evaluated over {num_episodes} episodes, average reward: {np.mean(rewards_all_episodes):.2f}, average steps per episode: {steps_taken / num_episodes:.2f}")
        return {
            "mean_reward": np.mean(rewards_all_episodes),
            "successful_episodes": successful_episodes,
            "num_episodes": num_episodes,
            "success_rate": successful_episodes / num_episodes,
            "max_steps_per_episode": max_steps_per_episode,
            "average_steps_per_episode": steps_taken / num_episodes,
        }

    def settings(self) -> dict:
        return {
            "num_episodes": self.num_episodes,
            "max_steps_per_episode": self.max_steps_per_episode,
            "learning_rate": self.learning_rate,
            "discount_rate": self.discount_rate,
            "exploration_rate": self.exploration_rate,
            "max_exploration_rate": self.max_exploration_rate,
            "min_exploration_rate": self.min_exploration_rate,
            "exploration_decay_rate": self.exploration_decay_rate,
        }

class PolicyIterationAlgorithm(Algorithm):
    """
    Policy Iteration Algorithm (model-based).
    Consists of two main steps:
    1. Policy Evaluation
        where we evaluate the current policy by calculating the value function V(s)

    2. Policy Improvement
        where we improve the policy based on the current value function V(s)
        policy(s) = argmax_a Q(s,a)
    """
    def __init__(
        self,
        discount_rate: float,
        max_iterations: int,
        evaluation_episodes: int = 100,
    ):
        super(PolicyIterationAlgorithm, self).__init__(AlgorithmType.POLICY_ITERATION)
        self.discount_rate = discount_rate
        self.max_iterations = max_iterations
        self.evaluation_episodes = evaluation_episodes
        self.q_table = None
        self.value_function = None
        self.history = []

        # evaluation metrics
        self.avg_rewards_history = []

    def _simulate_policy(self, env: gym.Env, num_eval_episodes: int = 50):
        total = 0
        max_steps_per_episode = 100
        for _ in range(num_eval_episodes):
            state , info = env.reset()
            done = False
            rsum = 0
            steps = 0
            while not done and steps < max_steps_per_episode:
                action = np.argmax(self.q_table[state]) 
                state,reward,terminated,truncated,_ = env.step(action)
                rsum += reward
                done = terminated or truncated
                steps += 1
            total += rsum
        return total / num_eval_episodes

    def train(
        self,
        env: gym.Env
    ):
        # initialize a random policy (i.e., Q-table with equal probabilities for each action)
        # in other algorithms we initialize the Q-table with zeros the reason we do it here with equal probabilities
        # is that in policy iteration we need a valid policy to start with
        self.q_table = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
        self.value_function = np.zeros(env.observation_space.n)

        # Policy Iteration algorithm steps:
        # 1. Policy Evaluation
        # 2. Policy Improvement

        for iteration in range(self.max_iterations):

            # Policy Evaluation step: update the value function based on the current policy
            for state in range(env.observation_space.n):
                new_value = 0
                for action in range(env.action_space.n):
                    for prob, next_state, reward, done in env.P[state][action]:
                        new_value += self.q_table[state, action] * prob * (reward + self.discount_rate * self.value_function[next_state])
                self.value_function[state] = new_value

            # Policy Improvement step: update the policy based on the current value function
            for state in range(env.observation_space.n):
                q_values = np.zeros(env.action_space.n)
                for action in range(env.action_space.n):
                    for prob, next_state, reward, done in env.P[state][action]:
                        q_values[action] += prob * (reward + self.discount_rate * self.value_function[next_state])
                best_action = np.argmax(q_values)
                # Update Q-table to be greedy w.r.t. the current value function
                self.q_table[state, :] = 0
                self.q_table[state, best_action] = 1

            self.history.append(self.q_table.copy())

            if iteration % 10 == 0:
                avg_reward = self._simulate_policy(env, num_eval_episodes=self.evaluation_episodes)
                self.avg_rewards_history.append(avg_reward)
        
        Utils.plot_q_value_convergence(
            np.array(self.history), 
            algorithm=self.algorithm_type
        )
        Utils.plot_avg_rewards(
            self.avg_rewards_history,
            window_size=min(self.evaluation_episodes // 10, len(self.avg_rewards_history)), 
            algorithm=self.algorithm_type
        )

    def evaluate(
        self,
        env: gym.Env,
        num_episodes: int,  
        max_steps_per_episode: int,
        time_sleep_between_episodes: float = 0.5,
        time_sleep_between_steps: float = 0.1,
        goal_reward: float = 1.0,
    ):
        if self.q_table is None:
            raise ValueError("The Q-table is not initialized. Please train the agent before evaluation.")

        rewards_all_episodes = []
        successful_episodes = 0
        steps_taken = 0

        for episode in range(num_episodes):
            state, info = env.reset()
            done = False
            print(f"*****EPISODE {episode + 1} *****")
            time.sleep(time_sleep_between_episodes)

            rewards_current_episode = 0
            for step in range(max_steps_per_episode):
                time.sleep(time_sleep_between_steps)
        
                action = np.argmax(self.q_table[state, :])
                new_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                rewards_current_episode += reward
                steps_taken += 1
                if done:
                    if reward == goal_reward:
                        print("****\tYou reached the goal!\t****")
                        time.sleep(time_sleep_between_episodes)
                        successful_episodes += 1
                    else:
                        print("****\tYou failed!\t****")
                        time.sleep(time_sleep_between_episodes)
                    break

                if step == max_steps_per_episode - 1:
                    print("****\tMaximum steps reached!\t****")
                    time.sleep(time_sleep_between_episodes)

                state = new_state
            rewards_all_episodes.append(rewards_current_episode)
        
        print(f"Evaluated over {num_episodes} episodes, average reward: {np.mean(rewards_all_episodes):.2f}, average steps per episode: {steps_taken / num_episodes:.2f}")
        return {
            "mean_reward": np.mean(rewards_all_episodes), 
            "successful_episodes": successful_episodes,
            "num_episodes": num_episodes,
            "success_rate": successful_episodes / num_episodes,
            "max_steps_per_episode": max_steps_per_episode,
            "average_steps_per_episode": steps_taken / num_episodes,
        }

    def settings(self) -> dict:
        return {
            "discount_rate": self.discount_rate,
            "max_iterations": self.max_iterations,
        }

class QValueIterationAlgorithm(Algorithm):
    """
    Q-Value Iteration Algorithm (model-based).
    Very similar to Value Iteration, but instead of updating state values (V),
    it updates state-action values (Q) directly using the Bellman optimality equation for Q-values.
    
    compared to Q-learning, Q-Value Iteration is a model-based method
    which means it requires knowledge of the environment's dynamics (transition probabilities and rewards)
    """
    def __init__(
        self,
        discount_rate: float,
        max_iterations: int,
        theta: float,
        evaluation_episodes: int = 100,
    ):
        super(QValueIterationAlgorithm, self).__init__(AlgorithmType.VALUE_ITERATION)
        self.discount_rate = discount_rate
        self.max_iterations = max_iterations
        self.theta = theta
        self.q_table = None
        self.evaluation_episodes = evaluation_episodes

        self.history = []
        self.avg_rewards_history = []

    def _simulate_policy(self, env: gym.Env, num_eval_episodes: int = 50):
        total = 0
        # NOTE: might get stuck in an infinite loop if the policy is bad
        # so we limit the number of steps per episode
        max_steps_per_episode = 100
        for n in range(num_eval_episodes):
            state, info = env.reset()
            done = False
            rsum = 0
            steps = 0
            while not done and steps < max_steps_per_episode:
                a = np.argmax(self.q_table[state]) 
                state, reward, terminated, truncated, _ = env.step(a)
                rsum += reward 
                done = terminated or truncated
                steps += 1
            total += rsum
        return total / num_eval_episodes

    def train(
        self,
        env: gym.Env
    ):
        nS = env.observation_space.n
        nA = env.action_space.n

        self.q_table = np.zeros((nS, nA))

        for iteration in range(self.max_iterations):
            delta = 0
            self.history.append(self.q_table.copy())
            for s in range(nS):
                for a in range(nA):
                    q_old = self.q_table[s, a]
                    q_new = 0
                    for prob, s2, reward, done in env.P[s][a]:
                        q_new += prob * (reward + self.discount_rate * np.max(self.q_table[s2]))

                    self.q_table[s, a] = q_new
                    delta = max(delta, abs(q_old - q_new))

            if iteration % 10 == 0:
                avg_reward = self._simulate_policy(env, num_eval_episodes=self.evaluation_episodes)
                self.avg_rewards_history.append(avg_reward)

            if delta < self.theta:
                break
            
        Utils.plot_q_value_convergence(
            np.array(self.history), 
            algorithm=self.algorithm_type
        )
        Utils.plot_avg_rewards(
            self.avg_rewards_history, 
            # if we have less data points than evaluation episodes, we adjust the window size accordingly
            window_size=min(self.evaluation_episodes // 10, len(self.avg_rewards_history)), 
            algorithm=self.algorithm_type
        )

    def evaluate(
        self,
        env: gym.Env,
        num_episodes: int,  
        max_steps_per_episode: int,
        time_sleep_between_episodes: float = 0.5,
        time_sleep_between_steps: float = 0.1,
        goal_reward: float = 1.0,
    ):
        if self.q_table is None:
            raise ValueError("The Q-table is not initialized. Please train the agent before evaluation.")
        
        rewards_all_episodes = []
        successful_episodes = 0
        steps_taken = 0

        for episode in range(num_episodes):
            state, info = env.reset()
            done = False
            print(f"*****EPISODE {episode + 1} *****")
            time.sleep(time_sleep_between_episodes)

            rewards_current_episode = 0
            for step in range(max_steps_per_episode):
                time.sleep(time_sleep_between_steps)
        
                action = np.argmax(self.q_table[state, :])
                new_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                rewards_current_episode += reward
                steps_taken += 1
                if done:
                    if reward == goal_reward:
                        print("****\tYou reached the goal!\t****")
                        time.sleep(time_sleep_between_episodes)
                        successful_episodes += 1
                    else:
                        print("****\tYou failed!\t****")
                        time.sleep(time_sleep_between_episodes)
                    break

                if step == max_steps_per_episode - 1:
                    print("****\tMaximum steps reached!\t****")
                    time.sleep(time_sleep_between_episodes)

                state = new_state
            rewards_all_episodes.append(rewards_current_episode)

        print(f"Evaluated over {num_episodes} episodes, average reward: {np.mean(rewards_all_episodes):.2f}, average steps per episode: {steps_taken / num_episodes:.2f}")
        return {
            "mean_reward": np.mean(rewards_all_episodes), 
            "successful_episodes": successful_episodes,
            "num_episodes": num_episodes,
            "success_rate": successful_episodes / num_episodes,
            "max_steps_per_episode": max_steps_per_episode,
            "average_steps_per_episode": steps_taken / num_episodes,
        }
    
    def settings(self) -> dict:
        return {
            "discount_rate": self.discount_rate,
            "max_iterations": self.max_iterations,
            "theta": self.theta,
        }

class MCMCAlgorithm(Algorithm):
    """
    Markov Chain Monte Carlo (MCMC) Algorithm implementation.
    Monte Carlo method derives its name from a Monte Carlo casino in Monaco.
    It is a technique for sampling from a probability distribution and using those samples 
    to approximate desired quantity. 
    In other words, it uses randomness to estimate some deterministic 
    quantity of interest.

    e.g.: estimating the area of a complex shape by randomly sampling points
    within a bounding box and determining the proportion that falls within the shape.
    This very much related to the central limit theorem in statistics.

    Original Expectation to be calculated:
        s = E_p[f(x)] = \int p(x) * f(x) dx

    Approximated Expectation generated by stimulating large samples of f(x): 
        E[X] ~ s ~ (1/N) * \sum_(i=1)^N f(x_i) where x_i ~ p(x)

    however, in many cases, directly sampling from p(x) is challenging.
    MCMC methods provide a way to generate samples from p(x) by constructing a Markov chain 
    (system that exhibits the Markov property "the future is independent of the past given the present")
    that has p(x) as its equilibrium distribution. By simulating this Markov chain over time (i.e., generating a sequence of samples),
    we can obtain samples that approximate the desired distribution.
    """
    def __init__(
        self,
        num_episodes: int,
        max_steps_per_episode: int,
        discount_rate: float,
        learning_rate: float = 0.1,
        exploration_rate: float = 1.0,
        max_exploration_rate: float = 1.0,
        min_exploration_rate: float = 0.01,
        exploration_decay_rate: float = 0.001,
    ):
        super(MCMCAlgorithm, self).__init__(AlgorithmType.MCMC)
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.max_exploration_rate = max_exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.q_table = None

        self.rewards_all_episodes = []
        self.history = []

    def train(
        self,
        env: gym.Env
    ):
        gen = np.random.default_rng(RANDOM_SEED)
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        # MCMC is a sampling-based method, which first generates episodes
        # and then uses these episodes to estimate value functions. compared to algorithms like Q-learning or DQN,
        # which update value estimates incrementally after each step.

        # MCMC:
        # Initialize value estimates
        # Q(s,a) = 0 for all s in S, a in A(s)
        # policy pi(a|s) = initial policy (e.g., random policy)
        #
        #
        # for each episode:
        #   generate an episode using the current policy
        #   initialize return (G = 0)  
        #   for each step in the episode:
        #       calculate the return G from that step onward
        #       G_t = gamma * G_(t+1) + r_t (reward at time t)
        #       update Q-value (policy) for (state_t, action_t) pair  
        #       using the bellman equation:
        #       Q(s,a) = (1 - alpha) * Q(s,a) + alpha * [R + gamma * max Q(s',a)] 

        for episode in range(self.num_episodes):
            state, info = env.reset()
            done = False

            # generate an episode using the current policy (the policy here is epsilon-greedy)
            # each time we sample an episode we get a different sequence of states
            # that the agent takes from the state being evaluated to the terminal state
            # based upon the transition matrix's probabilities
            episode_data = []
            episode_return = 0
            while not done and len(episode_data) < self.max_steps_per_episode:

                if gen.uniform(0,1) > self.exploration_rate:
                    action = np.argmax(self.q_table[state,:])
                else:
                    action = env.action_space.sample()

                new_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_return += reward
                episode_data.append((state, action, reward))
                state = new_state
                if done:
                    break

            self.rewards_all_episodes.append(episode_return)
            self.history.append(self.q_table.copy())
            # after generating the episode, we calculate the cummulative discounted reward G_t
            # from each time step t to the end of the episode using a finite horizon
            # G_t = R_t + gamma * R_(t+1) + gamma^2 * R_(t+2) + ... + gamma^(T-t) * R_T
            # <=> G_t = R_t + gamma * G_(t+1)
            G_t = 0 
            for state_t, action_t, reward_t in reversed(episode_data):
                G_t = reward_t + self.discount_rate * G_t
                # update Q-value for (state_t, action_t) pair   
                # <=> Q(s,a) = (1 - alpha) * Q(s,a) + alpha * [R + gamma * max Q(s',a)] 
                # <=> Q(s,a) = Q(s,a) - alpha * Q(s,a)  + alpha * [R + gamma * max Q(s',a)] 
                # <=> Q(s,a) = Q(s,a) + alpha * [R + gamma * max Q(s',a) - Q(s,a)]
                self.q_table[state_t, action_t] += self.learning_rate * (G_t - self.q_table[state_t, action_t])

            # Exploration rate decay update
            self.exploration_rate = self.min_exploration_rate + (self.max_exploration_rate - self.min_exploration_rate) * np.exp(-self.exploration_decay_rate * episode)

        Utils.plot_q_value_convergence(
            np.array(self.history), 
            algorithm=self.algorithm_type
        )
        Utils.plot_avg_rewards(
            self.rewards_all_episodes, 
            window_size=100, 
            algorithm=self.algorithm_type
        )

    def evaluate(
        self,
        env: gym.Env,
        num_episodes: int,
        max_steps_per_episode: int,
        time_sleep_between_episodes: float = 0.5,
        time_sleep_between_steps: float = 0.1,
        goal_reward: float = 1.0,
    ):
        if self.q_table is None:
            raise ValueError("The Q-table is not initialized. Please train the agent before evaluation.")

        rewards_all_episodes = []
        successful_episodes = 0
        steps_taken = 0
        
        for episode in range(num_episodes):
            state, info = env.reset()
            done = False
            print(f"*****EPISODE {episode + 1} *****")
            time.sleep(time_sleep_between_episodes)

            rewards_current_episode = 0
            for step in range(max_steps_per_episode):
                time.sleep(time_sleep_between_steps)
        
                action = np.argmax(self.q_table[state, :])
                new_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                rewards_current_episode += reward
                steps_taken += 1
                if done:
                    if reward == goal_reward:
                        print("****\tYou reached the goal!\t****")
                        time.sleep(time_sleep_between_episodes)
                        successful_episodes += 1
                    else:
                        print("****\tYou failed!\t****")
                        time.sleep(time_sleep_between_episodes)
                    break

                if step == max_steps_per_episode - 1:
                    print("****\tMaximum steps reached!\t****")
                    time.sleep(time_sleep_between_episodes)

                state = new_state
            rewards_all_episodes.append(rewards_current_episode)
        print(f"Average reward over {num_episodes} episodes: {np.mean(rewards_all_episodes)}, average steps per episode: {steps_taken / num_episodes:.2f}   ")
        
        return {
            "mean_reward": np.mean(rewards_all_episodes), 
            "successful_episodes": successful_episodes,
            "num_episodes": num_episodes,
            "success_rate": successful_episodes / num_episodes,
            "max_steps_per_episode": max_steps_per_episode,
            "average_steps_per_episode": steps_taken / num_episodes,
        }

    def settings(self) -> dict:
        return {
            "num_episodes": self.num_episodes,
            "max_steps_per_episode": self.max_steps_per_episode,
            "discount_rate": self.discount_rate,
            "learning_rate": self.learning_rate,
            "exploration_rate": self.exploration_rate,
            "max_exploration_rate": self.max_exploration_rate,
            "min_exploration_rate": self.min_exploration_rate,
            "exploration_decay_rate": self.exploration_decay_rate,
        }
    

class SARSAAlgorithm(Algorithm):
    """
    SARSA Algorithm (on-policy) 
    using epsilon-greedy policy with decaying exploration rate.
    """
    def __init__(
        self,
        num_episodes: int = 1000,
        max_steps_per_episode: int = 100,
        learning_rate: float = 0.001,
        discount_rate: float = 0.99,
        exploration_rate: float = 1.0, 
        max_exploration_rate: float = 1.0, 
        min_exploration_rate: float = 0.01,
        exploration_decay_rate: float = 0.001,
    ):
        super(SARSAAlgorithm, self).__init__(AlgorithmType.SARSA)
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.exploration_rate = exploration_rate
        self.max_exploration_rate = max_exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.q_table = None
        self.history = []
        # evaluation metrics
        self.rewards_all_episodes = []
        self.q_values_history = []

    def train(
        self,
        env: gym.Env
    ):
        pass 

    def evaluate(
        self,
        env: gym.Env,
        num_episodes: int,
        max_steps_per_episode: int,
        time_sleep_between_episodes: float = 0.5,
        time_sleep_between_steps: float = 0.1,
        goal_reward: float = 1.0,
    ):
        pass 

    def settings(self) -> dict:
        pass 

class ActorCriticAlgorithm(Algorithm):
    """
    Actor-Critic Algorithm implementation.
    Combines both policy-based and value-based methods.
    The actor is responsible for selecting actions based on the current policy,
    while the critic evaluates the actions taken by the actor by estimating the value function.
    """
    def __init__(
        self,
        num_episodes: int,
        max_steps_per_episode: int,
        learning_rate_actor: float,
        learning_rate_critic: float,
        discount_rate: float,
        exploration_rate: float = 1.0,
        max_exploration_rate: float = 1.0,
        min_exploration_rate: float = 0.01,
        exploration_decay_rate: float = 0.001,
        replay_buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_frequency: int = 10,
        warmup_steps: int = 1000,
    ):
        super(ActorCriticAlgorithm, self).__init__(AlgorithmType.ACTOR_CRITIC)
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.discount_rate = discount_rate
        self.exploration_rate = exploration_rate
        self.max_exploration_rate = max_exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.warmup_steps = warmup_steps

    def train(
        self,
        env: gym.Env
    ):
        pass 

    def evaluate(
        self,
        env: gym.Env,
        num_episodes: int,
        max_steps_per_episode: int,
        time_sleep_between_episodes: float = 0.5,
        time_sleep_between_steps: float = 0.1,
        goal_reward: float = 1.0,
    ):
        pass 

    def settings(self) -> dict:
        return {
            "num_episodes": self.num_episodes,
            "max_steps_per_episode": self.max_steps_per_episode,
            "learning_rate_actor": self.learning_rate_actor,
            "learning_rate_critic": self.learning_rate_critic,
            "discount_rate": self.discount_rate,
            "exploration_rate": self.exploration_rate,
            "max_exploration_rate": self.max_exploration_rate,
            "min_exploration_rate": self.min_exploration_rate,
            "exploration_decay_rate": self.exploration_decay_rate,
            "replay_buffer_size": self.replay_buffer_size,
            "batch_size": self.batch_size,
            "target_update_frequency": self.target_update_frequency,
            "warmup_steps": self.warmup_steps,
        }
    
class MCPolicyEvaluationAlgorithm(Algorithm):
    """
    Monte Carlo Policy Evaluation Algorithm implementation.
    Evaluates a given policy by estimating the value function V(s)
    using Monte Carlo sampling.
    To sample episodes, the algorithm follows the given policy:
    (e.g., epsilon-greedy policy derived from a Q-table or a stochastic policy matrix).

    what's different here compared to MCMC is that we are evaluating a given policy
    rather than learning a policy from scratch , hence the name policy evaluation.

    """
    def __init__(
        self,
        discount_rate: float,
        num_episodes: int,
    ):
        super(MCPolicyEvaluationAlgorithm, self).__init__(AlgorithmType.MC_POLICY_EVALUATION)
        self.discount_rate = discount_rate
        self.num_episodes = num_episodes
        self.value_function = None
        self.history = []

    def train(
        self,
        env: gym.Env,
        policy: np.ndarray
    ):
        if policy.shape != (env.observation_space.n, env.action_space.n):
            raise ValueError(f"Policy shape {policy.shape} does not match environment dimensions {(env.observation_space.n, env.action_space.n)}")  

        nS = env.observation_space.n
        self.value_function = np.zeros(nS)
        # N(s) = 0  (initialize counter for state s) for all s in S
        visited_states = np.zeros(nS)

        # Monte Carlo Policy Evaluation algorithm steps:
        #  G(s) = 0  (initialize total return for state s) for all s in S
        #  N(s) = 0  (initialize counter for state s) for all s in S
        # for i in range(num_iterations):
        #  generate an episode using the given policy
        #  G_i_t = r_i_t + gamma * r_i_(t+1) + gamma^2 * r_i_(t+2) + ... (calculate returns for each time step t in episode i)
        #  for each step in the episode:
        #    if this is the first time t that state s_t is visited in the episode i
        #       increment counter of total first visits: N(s) = N(s) + 1
        #       increment total return G(s) = G(s) + G_t
        #       update value function V(s) = G(s) / N(s)

        for episode in range(self.num_episodes):
            state, info = env.reset()
            done = False
            episode_data = []
            while not done:
                action = np.random.choice(np.arange(env.action_space.n), p=policy[state])
                new_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_data.append((state, reward))
                state = new_state
                if done:
                    break

            first_visit = set()
            # calculate returns G_t for each time step t in the episode
            G = 0
            for state_t, reward_t in reversed(episode_data):
                G = reward_t + self.discount_rate * G
                
                if state_t not in first_visit:
                    first_visit.add(state_t)
                    # N(s) = N(s) + 1
                    visited_states[state_t] += 1
                    # G(s) = G(s) + G_t
                    self.value_function[state_t] += G
                    # update value function V(s) = G(s) / N(s)
                    self.value_function[state_t] /= visited_states[state_t]

            self.history.append(self.value_function.copy())

    def evaluate(
        self,
        env: gym.Env,
        num_episodes: int,
        max_steps_per_episode: int,
        time_sleep_between_episodes: float = 0.5,
        time_sleep_between_steps: float = 0.1,
        goal_reward: float = 1.0,
    ):
        if self.value_function is None:
            raise ValueError("The value function is not initialized. Please train the agent before evaluation.")

        rewards_all_episodes = []
        successful_episodes = 0
        steps_taken = 0

        for episode in range(num_episodes):
            state, info = env.reset()
            done = False
            print(f"*****EPISODE {episode + 1} *****")
            time.sleep(time_sleep_between_episodes)

            rewards_current_episode = 0
            for step in range(max_steps_per_episode):
                time.sleep(time_sleep_between_steps)
        
                action = np.argmax(self.value_function[state, :])
                new_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                rewards_current_episode += reward
                steps_taken += 1
                if done:
                    if reward == goal_reward:
                        print("****\tYou reached the goal!\t****")
                        time.sleep(time_sleep_between_episodes)
                        successful_episodes += 1
                    else:
                        print("****\tYou failed!\t****")
                        time.sleep(time_sleep_between_episodes)
                    break

                if step == max_steps_per_episode - 1:
                    print("****\tMaximum steps reached!\t****")
                    time.sleep(time_sleep_between_episodes)

                state = new_state
            rewards_all_episodes.append(rewards_current_episode)
        print(f"Evaluated over {num_episodes} episodes, average reward: {np.mean(rewards_all_episodes):.2f}, average steps per episode: {steps_taken / num_episodes:.2f}")
        return {
            "mean_reward": np.mean(rewards_all_episodes), 
            "successful_episodes": successful_episodes,
            "num_episodes": num_episodes,
            "success_rate": successful_episodes / num_episodes,
            "max_steps_per_episode": max_steps_per_episode,
            "average_steps_per_episode": steps_taken / num_episodes,
        }
    
    def settings(self) -> dict:
        return {
            "discount_rate": self.discount_rate,
            "num_episodes": self.num_episodes,
        }


class MCIncrementalPolicyEvaluationAlgorithm(Algorithm):
    """
    Monte Carlo Incremental Policy Evaluation Algorithm implementation.
    Evaluates a given policy by estimating the value function V(s)
    using incremental updates based on Monte Carlo sampling.
    To sample episodes, the algorithm follows the given policy:
    (e.g., epsilon-greedy policy derived from a Q-table or a stochastic policy matrix).
    what's different here compared to MCPolicyEvaluationAlgorithm is that we are using
    incremental updates to the value function after each episode rather than averaging returns over all visits.
    """
    def __init__(
        self,
        discount_rate: float,
        learning_rate: float,
        num_episodes: int,
    ):
        super(MCIncrementalPolicyEvaluationAlgorithm, self).__init__(AlgorithmType.INCREMENTAL_MC_POLICY_EVALUATION)
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.num_episodes = num_episodes
        self.value_function = None
        self.history = []
        self.visit_counts = None

    def train(
        self,
        env: gym.Env,
        policy: np.ndarray
    ):
        if policy.shape != (env.observation_space.n, env.action_space.n):
            raise ValueError(f"Policy shape {policy.shape} does not match environment dimensions {(env.observation_space.n, env.action_space.n)}")  
        nS = env.observation_space.n
        self.value_function = np.zeros(nS)
        self.visit_counts = np.zeros(nS)

        # after each episode i = s1, a1, r1, s2, a2, r2, ..., sT
        # define return G_t = r_t + gamma * r_(t+1) + gamma^2 * r_(t+2) + ... + gamma^(T-t) * r_T
        # for state s visited at time t in episode i
        # increment counter of total visits: N(s) = N(s) + 1
        # update estimate 
        # V_pi(s) = V_pi(s) * (N(s) - 1) / N(s) + G_t / N(s)
        # <=> V_pi(s) = V_pi(s) + (1 / N(s)) * [G_t - V_pi(s)]
        pass

    def evaluate(
        self,
        env: gym.Env,
        num_episodes: int,
        max_steps_per_episode: int,
        time_sleep_between_episodes: float = 0.5,
        time_sleep_between_steps: float = 0.1,
        goal_reward: float = 1.0,
    ):
        if self.value_function is None:
            raise ValueError("The value function is not initialized. Please train the agent before evaluation.")

        rewards_all_episodes = []
        successful_episodes = 0
        steps_taken = 0

        for episode in range(num_episodes):
            state, info = env.reset()
            done = False
            print(f"*****EPISODE {episode + 1} *****")
            time.sleep(time_sleep_between_episodes)

            rewards_current_episode = 0
            for step in range(max_steps_per_episode):
                time.sleep(time_sleep_between_steps)
        
                action = np.argmax(self.value_function[state, :])
                new_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                rewards_current_episode += reward
                steps_taken += 1
                if done:
                    if reward == goal_reward:
                        print("****\tYou reached the goal!\t****")
                        time.sleep(time_sleep_between_episodes)
                        successful_episodes += 1
                    else:
                        print("****\tYou failed!\t****")
                        time.sleep(time_sleep_between_episodes)
                    break

                if step == max_steps_per_episode - 1:
                    print("****\tMaximum steps reached!\t****")
                    time.sleep(time_sleep_between_episodes)

                state = new_state
            rewards_all_episodes.append(rewards_current_episode)
        print(f"Evaluated over {num_episodes} episodes, average reward: {np.mean(rewards_all_episodes):.2f}, average steps per episode: {steps_taken / num_episodes:.2f}")
        return {
            "mean_reward": np.mean(rewards_all_episodes), 
            "successful_episodes": successful_episodes,
            "num_episodes": num_episodes,
            "success_rate": successful_episodes / num_episodes,
            "max_steps_per_episode": max_steps_per_episode,
            "average_steps_per_episode": steps_taken / num_episodes,
        }
    
    def settings(self) -> dict:
        return {
            "discount_rate": self.discount_rate,
            "learning_rate": self.learning_rate,
            "num_episodes": self.num_episodes,
        }
    
