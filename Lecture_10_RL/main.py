import matplotlib.pyplot as plt
import pygame
# NOTE: Make sure to install gymnasium package and not gym package
# gym has been deprecated 
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv

from lib.algorithms import *
from lib.global_const import *

# very good book on Reinforcement Learning:
# https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf
# SOURCES:
# ========
# https://www.researchgate.net/publication/235004620_Reinforcement_Learning_and_Markov_Decision_Processes
# https://medium.com/swlh/the-map-of-artificial-intelligence-2020-2c4f446f4e43
# https://medium.com/@souptik.reach.095/understanding-markov-reward-process-and-markov-decision-process-91f736a1b75c
# https://medium.com/intro-to-artificial-intelligence/the-actor-critic-reinforcement-learning-algorithm-c8095a655c14
# https://www.geeksforgeeks.org/what-is-markov-decision-process-mdp-and-its-relevance-to-reinforcement-learning/
# https://www.geeksforgeeks.org/actor-critic-algorithm-in-reinforcement-learning/
# https://www.appliedaicourse.com/blog/markov-decision-process-mdp/
# https://www.youtube.com/watch?v=RmOdTQYQqmQ
# https://ai.stackexchange.com/questions/42786/what-is-the-relation-between-dynamic-programming-and-reinforcement-learning
# https://www.geeksforgeeks.org/maths/linear-programming/
# https://rushi-prajapati.medium.com/demystifying-dynamic-programming-and-linear-programming-from-the-perspective-of-ai-and-15c9d0af13ac
# https://en.wikipedia.org/wiki/Linear_programming
# https://rushi-prajapati.medium.com/demystifying-dynamic-programming-and-linear-programming-from-the-perspective-of-ai-and-15c9d0af13ac
# https://towardsdatascience.com/using-linear-programming-to-boost-your-reinforcement-learning-algorithms-994977665902/
# https://dev.to/exactful/using-the-cross-entropy-method-to-solve-frozen-lake-3cea
# https://gsverhoeven.github.io/post/frozenlake-qlearning-convergence/
# aurelien geron - Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 3rd Edition

# a MDP (Markov Decision Process) is a mathematical framework to encapsulate the RL (Reinforcement Learning) problem
# it is a extension of a MRP(Markov Reward Process) 
# which in turn is a extension of the MC(Markov Chain) 
# which are very similar to Finite State Machines(FSM) and Bahavior Trees(BT) 
# the main difference is that BT and FSM are deterministic and MC, MRP and MDP are stochastic
# MDP, MRP, MC are all based on the assumption that the next state is only dependent on the current state
# an thus dismissing the past: "the future is independent of the past given the present" - Markov Property

# ================
# MC  <S,T>
# ================
# where,
# -> S is the State space (all possible states) (finite or infinite)
# the states is a convenient way of representing the environment and discribes what is changing at each time.
# The rules on how these states change are called the "dynamics" of the environment (T(s,s') aka Transition function or Transition Matrix) 
# which can either be deterministic or stochastic. 
# the states itself can be either discrete or continuous.
# a MC is either episodic or a continuous task. 
#
# https://arxiv.org/abs/2304.09831
# e.g.:
# discrete   -> int's X,Y positions in  grid world
# continuous -> [speed , angular velocity , steering angle]
# stochastic -> weather conditions, traffic conditions, random events
# deterministic -> chess game, go game, tic tac toe game, rule based systems
# episodic   -> the agent starts from a initial state and ends in a absorbing state (terminal state) or after a fixed number of steps 
# continuous  -> the agent continues to interact with the environment indefinitely
#
# NOTE: T(s,s') == P(s'|s) = 1 -> T is said to be deterministic
#       T(s,s') == P(s'|s) < 1 -> T is said to be stochastic

# ================
# MRP <S,T,R,γ,(α)>
# ================
# where,
# -> R is the Reward function/matrix R(s) or R(s,s') which indicates how good a state is
# it is used to guide the agent's behavior it can be either deterministic or stochastic
# -> γ is the Discount factor (gamma) is a hyperparameter to represent the importance of future rewards
# 
# MRP's also introduce the concept of a "return" which is the cumulative reward 
# from the current state until h-steps into the future
# There are basically three models of optimality in the MDP, which are:
# - finite horizon  E[G_t] = E[ R_t + R_(t+1) + R_(t+2) + ... + R_(t+h) ] with t -> t+h
# - discounted infinite horizon E[G_t] = E[ R_t + γ*R_(t+1) + γ^2*R_(t+2) + ... ] with t -> infinity
# - average reward  E[G_t] = lim (h->inf) E[  1/h * (R_t + R_(t+1) + ... + R_(t+h-1)) ] with t -> h
#
# NOTE: h can be equal to infinity and gamma can be equal to 1 but not both at the same time 
#       otherwise the return G_t will be infinite
#
# NOTE: R(s,s') == constant -> R is said to be deterministic
#       R(s,s') == variable -> R is said to be stochastic
#
# NOTE: γ == 0 -> agent only cares about immediate rewards
#       γ == 1 -> agent cares about all future rewards
#
# NOTE: choosing appropriate rewards can be a tricky issue do to problems like 
#       "Reward hijacking" , "Temporal Credit Assignment problem" and "overshadowing" 
#
# furthermore MRPs also introduce the concept of a "value function" 
# V(s) which indicates how good it is for the agent to be in a given state
# by calculating the expected return from that state
# V(s) = E[G_t | S_t = s]
# <=> V(s) = E[ R_t + γ*R_(t+1) + γ^2*R_(t+2) + ... + γ^h*R_(t+h)  | S_t = s ]
# <=> V(s) = E[ R_t + γ*V(S_(t+1)) | S_t = s ]
# we know that E[x] = sum( p(x) * x )
# so we can rewrite the equation as:
# <=> V(s) = R(s) + γ * sum((for all states) T(s,s') * V(s'))
# because we are taking the expected value over all possible next states s'
# we don't know which state s' the agent will end up in so we take the "weighted" average over all possible next states s'
# this equation is called the Bellman equation for the value function in a MRP
# it will be guaranteed to converge to the optimal value function V* as long as the discount factor γ is less than 1
# MRPs can be solved in a few ways:
#
# 1) by iteration
#   initialize V(s) = 0 for all states s
#   repeat:
#       for each state s in S:
#           V(s) = R(s) + γ * sum((for all states) T(s,s') * V(s'))
#   until V converges to the optimal value function V*
#
# 2) by linear algebra 
# solving the value function by linear algebra:
#   the value function can be rewritten in matrix form as:
#   V = R + gamma * T * V
#   <=> V - gamma * T * V = R
#   <=> (I - gamma * T) * V = R 
#   <=> V = (I - gamma * T)^-1 * R
# where 
# I is the identity matrix 
#     
#     | P(s1|s1) P(s2|s1) ... P(sn|s1) |
#     | P(s1|s2) P(s2|s2) ... P(sn|s2) |
# T = |   ...       ...         ...    |
#     | P(s1|sn) P(s2|sn) ... P(sn|sn) |
#   
#  R = | R(s1) |   
#      | R(s2) |
#      |  ...  |
#      | R(sn) |
#
# V = | V(s1) |
#     | V(s2) |
#     |  ...  |
#     | V(sn) |
# 
#
# 3) by simulation (Monte Carlo)
#   initialize V(s) = 0 for all states s
#   for each episode:
#       generate an episode using the current policy
#       for each state s in the episode:
#           G_t = cumulative reward from state s
#           V(s) = V(s) + α * (G_t - V(s))
# where α is the learning rate (0 < α <= 1)
#
# NOTE: both methods require the transition matrix T to be known which in most real world scenarios is not the case
#       thus we introduce methods like Monte Carlo and Temporal Difference learning to estimate the value function without knowing T
# 
# NOTE: in the beginning of the training the value function V(s) is initialized to 0 for all states s

# ================
# MDP <S,A,T,R,γ,π,(ε),(α)>
# ================
# where,
# -> A is the Action space (all possible actions) (finite or infinite)
# the actions is what the agent can do to interact with the environment and change its state.
# in an MDP the agent chooses an action a from the action space A based on a policy 
# -> π(a|s) which is a mapping from states to actions the policy can be either deterministic or stochastic
# e.g.:
# deterministic -> π(a|s) == 1 (e.g.: rule based systems)
# stochastic    -> π(a|s) = P(a|s) (e.g.: probability distribution (uniform, Gaussian, etc.) over actions given state)
#
# NOTE:  alpha (α) is the learning rate used in Temporal Difference learning to update the value function 
#
# NOTE: epsilon (ε) is the exploration rate used in epsilon-greedy policies to balance exploration and exploitation 
#
# NOTE: in an MDP the transition function T(s,s') is replaced by T(s,a,s') which indicates 
#       the probability of transitioning from state s to state s' given action a
# 
# NOTE: in an MDP the reward function R(s) is replaced by R(s,a) or R(s,a,s') which indicates 
#       the reward received after taking action a in state s and transitioning to state s'
#
# NOTE: in an MDP the value function V(s) definition is modified to take into account the policy π
# V_π(s) = E[G_t | S_t = s, π]
# <=> V_π(s) = E[ R_t + γ*R_(t+1) + γ^2*R_(t+2) + ... + γ^h*R_(t+h)  | S_t = s, π ]
# <=> V_π(s) = E[ R_t + γ*V(S_(t+1)) | S_t = s, π ]
# again we know that E[x] = sum( p(x) * x )
# so we can rewrite the equation as:
# <=> V_π(s) = sum((for all actions) π(a|s) * [ R(s,a) + γ * sum((for all states) T(s,a,s') * V_π(s')) ])
# because we are taking the expected value over all possible actions a AND all possible next states s'
# we don't know which action a the agent will take or which state s' the agent will end up in
# so we take the "weighted" average over all possible actions a and next states s'
# this equation is called the Bellman equation for the value function in an MDP
#  
# MDPs also introduce the concept of a "Q-value function" 
# Q(s,a) which indicates how good it is for the agent to take a given action in a given state
# by calculating the expected return from that state and action
# Q_π(s,a) = E[G_t | S_t = s, A_t = a, π]
# <=> Q_π(s,a) = E[ R_t + γ*R_(t+1) + γ^2*R_(t+2) + ... + γ^h*R_(t+h)  | S_t = s, A_t = a, π ]
# <=> Q_π(s,a) = E[ R_t + γ*V_π(S_(t+1)) | S_t = s, A_t = a, π ]
# we can rewrite the equation as:
# <=> Q_π(s,a) = R(s,a) + γ * sum((for all states) T(s,a,s') * V_π(s'))
# this equation is called the Bellman equation for the Q-value function in an MDP
# 
# NOTE: the optimal value function V*(s) and optimal Q-value function Q*(s,a) can be derived from the Bellman equations
#       by taking the maximum over all policies π
#       V*(s) = max( for all π ) V_π(s)
#       Q*(s,a) = max( for all π ) Q_π(s,a)
#
# NOTE: in the beginning of the training the Q-value function Q(s,a) is initialized to 0 for all states s and actions a
#
# MDPs can be solved in two ways:
# 
# 1) by iteration
#   1.1) Policy Iteration
#       initialize a random policy π
#       repeat:
#           evaluate the value function V_π for the current policy π
#           improve the policy π by acting greedily with respect to V_π
#       until policy π converges to the optimal policy π*
#
#   1.2) Value Iteration
#   initialize V(s) = 0 for all states s
#   repeat:
#       for each state s in S:
#           V(s) = max( for all actions a ) [ R(s,a) + γ * sum((for all states) T(s,a,s') * V(s')) ]
#   until V converges to the optimal value function V*
#   derive the optimal policy π*
#   for each state s in S:
#       π*(s) = argmax( for all actions a ) [ R(s,a) + γ * sum((for all states) T(s,a,s') * V*(s'))
#
# 2) Monte Carlo and Temporal Difference learning
#   initialize Q(s,a) = 0 for all states s and actions a
#   for each episode:
#       generate an episode using the current policy π
#       for each state s and action a in the episode:
#           G_t = cumulative reward from state s and action a
#           Q(s,a) = Q(s,a) + α * (G_t - Q(s,a))
#   derive the optimal policy π*
#
# NOTE: both methods require the transition matrix T to be known which in most real world scenarios is not the case
#       thus we introduce methods like Q-learning and SARSA to estimate the Q-value function without knowing T
#       also called "model-free" methods
#
# NOTE: a good policy π is maybe not the optimal policy π* so how can now when we have a good policy?
#       we can use methods like "epsilon-greedy" or "softmax" to 
#       balance exploration (trying new actions) and exploitation (choosing the best known action)
#
#

# ================
# Algorithms in RL
# ================
# most algorithms in RL can be categorized into categories based on whether they are:
# 1) MODEL-BASED METHODS
#    Require transition model T(s,a,s') and reward model R(s,a)
#
#   1.1) Value-based methods (estimate value functions; gradient-free)
#        - Value Iteration
#        - Policy Iteration (has value-evaluation step)
#        - Q-Value Iteration (Bellman optimality on Q)
#        
#       Monte Carlo category inside model-based:
#       - MCMC Planning / MCMC Control (sampling-based policy evaluation or planning
#         using a known model; uses rollouts from the model instead of full Bellman updates)
#
#   1.2) Policy-based methods (optimize policies directly; gradient-based methods)
#       - Model-based Policy Search
#       - Model Predictive Control (MPC)
#       - iLQR / DDP (in continuous control)
#
# 2) MODEL-FREE METHODS
#    Do NOT require knowledge of T or R. Learn from experience only
#
#   2.1) Value-based methods (estimate V or Q; gradient-free)
#
#       2.1.1) Off-policy
#              - Q-Learning
#              - Expected Q-learning
#              - Double Q-learning
#              - Deep Q-Networks (DQN)
#              - Dueling DQN
#
#       2.1.2) On-policy
#              - SARSA
#              - Expected SARSA
#              - TD(0), TD(λ)  (temporal-difference learning)
#              - Monte Carlo Control (on-policy every-visit MC estimate of Q)
#              - Monte Carlo Policy Evaluation (on-policy V or Q estimation by averaging returns)
#
#   2.2) Policy-based methods (directly optimize πθ; gradient-based)
#
#       2.2.1) Policy Gradient Methods
#              - REINFORCE (Monte Carlo Policy Gradient)
#              - PPO (Proximal Policy Optimization)
#              - TRPO
#              - Soft Actor-Critic (in max-entropy setting, partly actor-critic)
#
#       2.2.2) Actor–Critic Methods
#              - A2C (Advantage Actor-Critic)
#              - A3C
#              - TD Actor–Critic
#              - PPO-Actor-Critic (same algorithmic family)
#              - DDPG, TD3 (deterministic actor-critic)
#              - SAC (stochastic actor-critic)
#
# 



def main():
    # https://gymnasium.farama.org/environments/toy_text/frozen_lake/#arguments
    # is_slippery=True: If true the player will move in intended direction 
    # with probability specified by the success_rate else will move in either perpendicular direction with equal probability in both directions.
    # For example, if action is left, is_slippery is True, and success_rate is 1/3, then:
    # P(move left)=1/3
    # P(move up)=1/3
    # P(move down)=1/3
    # If action is up, is_slippery is True, and success_rate is 3/4, then:
    # P(move up)=3/4
    # P(move left)=1/8
    # P(move right)=1/8
    # success_rate=1.0/3.0: Used to specify the probability of moving in the intended direction when is_slippery=True
    # reward_schedule=(1, 0, 0): Used to specify reward amounts for reaching certain tiles. The indices correspond to: Reach Goal, Reach Hole, Reach Frozen (includes Start), Respectively
    
    # is default to True in the frozen lake environment
    # NOTE: setting slippery to True makes the environment stochastic and thus much more challenging for the agent to learn
    # it's true that in real world scenarios the environment is mostly stochastic but 
    # for learning purposes it is better to start with a deterministic environment
    ENV_IS_SLIPPERY = False

    # (Reach Goal, Reach Hole, Reach Frozen)
    # we could have had a other heuristic reward function (here deterministic):
    # - euclidean distance to goal
    # - manhattan distance to goal
    # - negative reward for each step taken to encourage the agent to find the shortest path
    # or like in MDP's we can define the reward function/matrix as = R(s,a,s')
    ENV_GOAL_REWARD = 1
    ENV_REWARD_SCHEDULE = (ENV_GOAL_REWARD, -1, -0.01)
    # so basically with success_rate=0.8
    # if the agent wants to go in one direction
    # it will go in that direction with probability 0.8
    # and with probability (1 - 0.8)/2 = 0.1 it will go in either perpendicular direction
    #  0.1
    #   |
    # Agent -> 0.8
    #   |
    #  0.1
    ENV_SUCCESS_RATE = 0.8

    # defines the environment layout
    ENV_DESC_GENERATED = Utils.generate_frozen_lake_desc(
        width=60,
        height=60,
        hole_probability=0.2,
        random_seed=RANDOM_SEED,
    )
    ENV_DESC = ENV_DESC_GENERATED

    STEP_TIME_SLEEP = 0.0
    EPISODE_TIME_SLEEP = 0.0
    EVAL_EPISODES = 100

    algorithm_dqn = DQNAlgorithm(
        num_episodes=1500,
        max_steps_per_episode=150,
        learning_rate=0.001,
        discount_rate=0.9,
        exploration_rate=0.99,
        max_exploration_rate=0.99,
        min_exploration_rate=0.01,
        exploration_decay_rate=0.001,
        replay_buffer_size=10000,
        batch_size=64,
        target_update_frequency=10,
        warmup_steps=1000,
    )

    algorithm_q_learn = QLearningAlgorithm(
        num_episodes=100000,
        max_steps_per_episode=150,
        learning_rate=0.1,
        discount_rate=0.9,
        exploration_rate=0.99,
        max_exploration_rate=0.99,
        min_exploration_rate=0.01,
        exploration_decay_rate=0.001,
    )

    # model based algorithms don't need that many iterations to converge
    # value iteration and policy iteration are guaranteed to converge to the optimal policy
    # however they do require the transition and reward model of the environment
    # also the solution space grows exponentially with the number of states and actions
    # lower theta if you want to converge to a more accurate solution
    algorithm_q_iter = QValueIterationAlgorithm(
        max_iterations=100,
        discount_rate=0.8,
        theta=1e-10,
    )

    algorithm_policy_iter = PolicyIterationAlgorithm(
        max_iterations=100,
        discount_rate=0.8,
    )

    # monte carlo methods usually need more episodes 
    # to converge same as model-free methods
    algorithm_mcmc = MCMCAlgorithm(
        num_episodes=100000,
        max_steps_per_episode=150,
        learning_rate=0.001,
        discount_rate=0.9,
        exploration_rate=0.99,
        max_exploration_rate=0.99,
        min_exploration_rate=0.01,
        exploration_decay_rate=0.001,
    )

    algorithms = [
        # AlgorithmType.MCMC,
        # AlgorithmType.VALUE_ITERATION,
        # AlgorithmType.POLICY_ITERATION,
        AlgorithmType.Q_LEARNING,
        # AlgorithmType.DQN,
    ] 

    PLOT_RESULTS = False

    for algorithm in algorithms:
        for render_mode, mode_name in zip(
            ["rgb_array", None, None], 
            [ "Render Mode","Training Mode", "Evaluation Mode"]
        ):
            env = FrozenLakeEnv(
                render_mode=render_mode,
                desc=ENV_DESC,
                is_slippery=ENV_IS_SLIPPERY,
                reward_schedule=ENV_REWARD_SCHEDULE,
                success_rate=ENV_SUCCESS_RATE,
            )
            env.action_space.seed(RANDOM_SEED)
            env.observation_space.seed(RANDOM_SEED)

            if mode_name == "Training Mode":
                print(f"\nTraining with algorithm: {algorithm.value}\n")
                if algorithm == AlgorithmType.DQN:
                    algorithm_dqn.train(env)

                elif algorithm == AlgorithmType.Q_LEARNING:
                    algorithm_q_learn.train(env)
                    if PLOT_RESULTS:
                        Utils.plot_qtable_probabilities(
                            q_table=algorithm_q_learn.q_table,
                            algorithm=algorithm,
                            grid_shape=env.desc.shape,
                            temp=0.5,
                            min_prob_threshold=0.01,
                        )
                
                elif algorithm == AlgorithmType.VALUE_ITERATION:
                    algorithm_q_iter.train(env)
                    if PLOT_RESULTS:
                        Utils.plot_qtable_probabilities(
                            q_table=algorithm_q_iter.q_table,
                            algorithm=algorithm,
                            grid_shape=env.desc.shape,
                            temp=0.5,
                            min_prob_threshold=0.01,
                        )

                elif algorithm == AlgorithmType.POLICY_ITERATION:
                    algorithm_policy_iter.train(env)
                    if PLOT_RESULTS:
                        Utils.plot_qtable_probabilities(
                            q_table=algorithm_policy_iter.q_table,
                            algorithm=algorithm,
                            grid_shape=env.desc.shape,
                            temp=0.5,
                            min_prob_threshold=0.01,
                        )

                elif algorithm == AlgorithmType.MCMC:
                    algorithm_mcmc.train(env)
                    if PLOT_RESULTS:
                        Utils.plot_qtable_probabilities(
                            q_table=algorithm_mcmc.q_table,
                            algorithm=algorithm,
                            grid_shape=env.desc.shape,
                            temp=0.5,
                            min_prob_threshold=0.01,
                        )

            elif mode_name == "Evaluation Mode":
                print(f"\nEvaluating algorithm: {algorithm.value}\n")
                if algorithm == AlgorithmType.DQN:
                    eval_results = algorithm_dqn.evaluate(
                        env,
                        num_episodes=EVAL_EPISODES,
                        max_steps_per_episode=150,
                        time_sleep_between_episodes=EPISODE_TIME_SLEEP,
                        time_sleep_between_steps=STEP_TIME_SLEEP,
                        goal_reward=ENV_GOAL_REWARD,
                    )
                    Utils.save_settings(
                        settings=algorithm_dqn.settings(),
                        algorithm=algorithm,
                    )
                    Utils.save_results(
                        results=eval_results,
                        algorithm=algorithm,
                    )

                elif algorithm == AlgorithmType.Q_LEARNING:
                    eval_results = algorithm_q_learn.evaluate(
                        env,
                        num_episodes=EVAL_EPISODES,
                        max_steps_per_episode=150,
                        time_sleep_between_episodes=EPISODE_TIME_SLEEP,
                        time_sleep_between_steps=STEP_TIME_SLEEP,
                        goal_reward=ENV_GOAL_REWARD,
                    )
                    Utils.save_settings(
                        settings=algorithm_q_learn.settings(),
                        algorithm=algorithm,
                    )
                    Utils.save_results(
                        results=eval_results,
                        algorithm=algorithm,
                    )

                elif algorithm == AlgorithmType.VALUE_ITERATION:
                    eval_results = algorithm_q_iter.evaluate(
                        env,
                        num_episodes=EVAL_EPISODES,
                        max_steps_per_episode=150,
                        time_sleep_between_episodes=EPISODE_TIME_SLEEP,
                        time_sleep_between_steps=STEP_TIME_SLEEP,
                        goal_reward=ENV_GOAL_REWARD,
                    )
                    Utils.save_settings(
                        settings=algorithm_q_iter.settings(),
                        algorithm=algorithm,
                    )
                    Utils.save_results(
                        results=eval_results,
                        algorithm=algorithm,
                    )

                elif algorithm == AlgorithmType.POLICY_ITERATION:
                    eval_results = algorithm_policy_iter.evaluate(
                        env,
                        num_episodes=EVAL_EPISODES,
                        max_steps_per_episode=150,
                        time_sleep_between_episodes=EPISODE_TIME_SLEEP,
                        time_sleep_between_steps=STEP_TIME_SLEEP,
                        goal_reward=ENV_GOAL_REWARD,
                    )
                    Utils.save_settings(
                        settings=algorithm_policy_iter.settings(),
                        algorithm=algorithm,
                    )
                    Utils.save_results(
                        results=eval_results,
                        algorithm=algorithm,
                    )

                elif algorithm == AlgorithmType.MCMC:
                    eval_results = algorithm_mcmc.evaluate(
                        env,
                        num_episodes=EVAL_EPISODES,
                        max_steps_per_episode=150,
                        time_sleep_between_episodes=EPISODE_TIME_SLEEP,
                        time_sleep_between_steps=STEP_TIME_SLEEP,
                        goal_reward=ENV_GOAL_REWARD,
                    )
                    Utils.save_settings(
                        settings=algorithm_mcmc.settings(),
                        algorithm=algorithm,
                    )
                    Utils.save_results(
                        results=eval_results,
                        algorithm=algorithm,
                    )

            elif mode_name == "Render Mode": 
                # first reset the environment to get the initial frame/state
                env.reset()
                frame = env.render()
                Utils.save_env_canvas(frame)
                env.close()

                Utils.save_env_settings(
                    settings_env={
                        "ENV_DESC": ENV_DESC,
                        "ENV_IS_SLIPPERY": ENV_IS_SLIPPERY,
                        "ENV_REWARD_SCHEDULE": ENV_REWARD_SCHEDULE,
                        "ENV_SUCCESS_RATE": ENV_SUCCESS_RATE,
                    }
                )

            
if __name__ == "__main__":
    main()
