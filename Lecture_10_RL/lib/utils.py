from pathlib import Path

import json
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .algorithms import AlgorithmType
from .global_const import RESULTS_PATH

CUSTOM_MAPS = {
    "5x5" : [
        "SFFFF",
        "HHFHH",
        "FFFHH",
        "HFFFF",
        "HFFFG",
    ],
    "8x8" : [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
    "20x20" : [
        "SFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFF",
        "FFFHFFFFFFFFFFFFFFFFF",
        "FFFFFHFFFFFFFFFFFFFFF",
        "FFFHFFFFFFFFFFFFFFFFF",
        "FHHFFFFFFFFFFFFFFFFFF",
        "FHFFHFHFFFFFFFFFFFFFF",
        "FFFHFFFGFFFFFFFFFFFFF",
    ],
    "RAND" : [
        "SFFHFFFFFHFHFFHFFFHFH",
        "FHFFHFFFHFFHFHFFHFFFH",
        "FFFHFFFFFHFFFFFHFFFFH",
        "HFFFHFFHFFFHFFHFFFHFH",
        "FFHFFFFFHFFHFFFFFHFFH",
        "HFFHFFFHFFHFFFHFFHFFH",
        "FFFHFFFFFHFFFFFHFFFGH",
        "HFFFHFFHFFFHFFHFFFHFH",
        "FFHFFFFFHFFHFFFFFHFFH",
        "HFFHFFFHFFHFFFHFFHFFH",
        "FFFHFFFFFHFFFFFHFFFFH",
        "HFFFHFFHFFFHFFHFFFHFH",
        "FFHFFFFFHFFHFFFFFHFFH",
        "HFFHFFFHFFHFFFHFFHFFH",
        "FFFHFFFFFHFFFFFHFFFFH",
        "HFFFHFFHFFFHFFHFFFHFH",
        "FFHFFFFFHFFHFFFFFHFFH",
        "HFFHFFFHFFHFFFHFFHFFH",
        "FFFHFFFFFHFFFFFHFFFFH",
        "FFFFFFFFFFFFFFFFFFFFG",
    ]
}

class Utils:
    results_dir = Path(RESULTS_PATH) / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    
    @staticmethod
    def plot_avg_rewards(rewards_history, window_size=100, algorithm: AlgorithmType = None, title="Average Reward per Episode"):
        """ Plots the average reward per episode with a rolling window to smooth out fluctuations."""
        if Utils.results_dir.exists() is False:
            Utils.results_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(rewards_history).to_parquet(
            Utils.results_dir / f"{algorithm.value if algorithm else 'Algorithm'}_rewards_history.parquet", 
            index=True, 
        )        
        
        rewards_series = np.array(rewards_history)
        # use convolution to efficiently calculate the rolling mean
        # we use 'valid' mode to only return results where the window fully overlaps
        weights = np.ones(window_size) / window_size
        smoothed_rewards = np.convolve(rewards_series, weights, mode='valid')
        # the smoothed curve is shorter, so create a corresponding episode axis
        episodes = np.arange(window_size, len(rewards_series) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(rewards_series)), rewards_series, alpha=0.3, label='Raw Rewards')
        plt.plot(episodes, smoothed_rewards, color='blue', linewidth=2, label=f'Smoothed Avg (Window={window_size})')
        if algorithm:
            plt.title(f'{algorithm.value} {title}')
        else:
            plt.title(f'{title}')
        plt.xlabel('Episode')
        plt.ylabel(f'Cumulative Reward (Max={np.max(rewards_series):.2f})')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig(Utils.results_dir / f"{algorithm.value if algorithm else 'Algorithm'}_avg_rewards.svg", format='svg')
        plt.close()

    @staticmethod 
    def plot_qtable_probabilities(
        q_table: np.ndarray, 
        grid_shape: tuple[int, int], 
        title="Q-Table Action Likelihood (Text Symbols)", 
        algorithm: AlgorithmType = None, 
        temp: float = 0.5,
        min_prob_threshold: float = 0.05,
        max_font_size: int = 30
    ):
        """
        Plots the action likelihood for each state in a grid world based on the Q-table,
        using the SIZE of the arrow text symbol to represent the probability/likelihood.

        Args:
            q_table (np.ndarray): The Q-table of shape (num_states, num_actions).
            grid_shape (tuple[int, int]): The (rows, cols) of the grid environment.
            title (str): The main title for the plot.
            algorithm (AlgorithmType): The name of the algorithm used.
            temp (float): The temperature parameter for the Softmax calculation.
            min_prob_threshold (float): Arrows with probability below this threshold are not plotted.
            max_font_size (int): The maximum font size for the most likely action.
        """
        
        rows, cols = grid_shape
        num_states, num_actions = q_table.shape

        if num_states != rows * cols:
            raise ValueError(
                f"Q-table states ({num_states}) must match grid size ({rows} * {cols})."
            )
        if num_actions not in [4, 8]:
            raise NotImplementedError(
                "Function is designed for 4 or 8 actions (Up, Right, Down, Left, etc.)."
            )

        # 1. Setup and Mappings
        # Action mappings: Symbols
        action_symbols = {
            0: '←',   # Left
            1: '↓',   # Down
            2: '→',   # Right
            3: '↑'    # Up
        }
        
        # Define relative positions for the four action symbols within the cell
        # This helps separate the multiple action symbols
        relative_offsets = {
            # (dx, dy)
            0: (-0.2, 0),    # Left symbol moved slightly left
            1: (0, 0.2),     # Down symbol moved slightly down
            2: (0.2, 0),     # Right symbol moved slightly right
            3: (0, -0.2)     # Up symbol moved slightly up
        }
        
        # 2. Plot Setup
        fig, ax = plt.subplots(figsize=(cols * 1.5, rows * 1.5))
        
        if algorithm:
            full_title = f"{title}\n({algorithm}, Softmax Temperature: {temp})"
        else:
            full_title = title
        ax.set_title(full_title)

        # Use a uniform color background
        ax.imshow(np.zeros(grid_shape), cmap='gray_r', alpha=0.1, origin='upper')

        # 3. Iterate through States and Plot Text Symbols
        for state in range(num_states):
            r = state // cols  # Row coordinate
            c = state % cols   # Column coordinate
            
            q_values = q_table[state, :]
            
            # Check for the special case: all actions have Q=0
            if np.all(q_values == 0):
                ax.text(c, r, '?', ha='center', va='center', color='red', fontsize=20, fontweight='bold')
                continue

            # Calculate action probabilities using Softmax
            exp_q = np.exp(q_values / temp)
            probabilities = exp_q / np.sum(exp_q)
            
            # Determine max probability for scaling the font size
            max_prob = np.max(probabilities)
            
            # Plot arrows for all actions above the threshold
            for action in range(num_actions):
                prob = probabilities[action]
                
                if prob > min_prob_threshold:
                    symbol = action_symbols.get(action, '?')
                    dx, dy = relative_offsets.get(action, (0, 0))
                    
                    # Scale font size based on its probability relative to the max probability
                    # This ensures the font size is visually proportional to the likelihood
                    
                    # Formula: base_size + (scale * (prob / max_prob))
                    # base_size ensures even low-probability arrows are visible
                    base_size = max_font_size * 0.2
                    
                    # Scale factor based on probability
                    scale_factor = (prob / max_prob) 
                    
                    # Final font size
                    font_size = base_size + ((max_font_size - base_size) * scale_factor)

                    # Plot the symbol
                    ax.text(
                        c + dx, r + dy, symbol,
                        ha='center', va='center',
                        color='black', 
                        fontsize=font_size,
                        fontweight='bold',
                        alpha=0.8 # Slight transparency for better overlap visibility
                    )

        # 4. Final Formatting
        ax.set_xticks(np.arange(cols))
        ax.set_yticks(np.arange(rows))
        ax.set_xticklabels(np.arange(cols))
        ax.set_yticklabels(np.arange(rows))
        ax.set_xlabel("Column Index")
        ax.set_ylabel("Row Index")
        ax.set_aspect('equal')

        # Minor ticks for grid lines
        ax.set_xticks(np.arange(cols + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(rows + 1) - 0.5, minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
        ax.tick_params(which='minor', size=0)

        plt.tight_layout()
        plt.savefig(Utils.results_dir / f"{algorithm.value if algorithm else 'Algorithm'}_qtable_action_likelihoods.svg", format='svg')
        plt.savefig(Utils.results_dir / f"{algorithm.value if algorithm else 'Algorithm'}_qtable_action_likelihoods.png", format='png', dpi=300)
        plt.close()

    @staticmethod
    def plot_q_value_convergence(history, title="Q-Value Convergence",algorithm: AlgorithmType = None):
        """ Plots the convergence of Q-values over iterations. (basically delta between Q-values over iterations)"""
        # https://github.com/ageron/handson-ml3/blob/main/18_reinforcement_learning.ipynb
        if Utils.results_dir.exists() is False:
            Utils.results_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 4))
        if algorithm:
            plt.title(f'{algorithm.value} {title}')
        else:
            plt.title(title)
        plt.ylabel("Q-Value$(s_0, a_0)$", fontsize=14)
        plt.plot(np.arange(len(history)), history[:, 0, 0], "b-", linewidth=2)
        plt.xlabel("Iterations", fontsize=14)
        plt.grid(True)
        plt.savefig(Utils.results_dir / f"{algorithm.value if algorithm else 'Algorithm'}_q_value_convergence.svg", format='svg')
        plt.close()

    @staticmethod
    def save_settings(settings: dict, filename="settings.json",algorithm: AlgorithmType = None):
        if Utils.results_dir.exists() is False:
            Utils.results_dir.mkdir(parents=True, exist_ok=True)

        if algorithm:
            filename = f"{algorithm.value}_{filename}"

        with open(Utils.results_dir / filename, "w") as f:
            json.dump(settings, f, indent=4)

    @staticmethod
    def save_env_settings(settings_env: dict, filename="env_description.json"):
        if Utils.results_dir.exists() is False:
            Utils.results_dir.mkdir(parents=True, exist_ok=True)

        with open(Utils.results_dir / filename, "w") as f:
            json.dump(settings_env, f, indent=4)

    @staticmethod
    def save_env_canvas(frame, filename="env_map.png"):
        if Utils.results_dir.exists() is False:
            Utils.results_dir.mkdir(parents=True, exist_ok=True)

        plt.imsave(Utils.results_dir / filename, frame)

    @staticmethod
    def save_results(results: dict, filename="results.json",algorithm: AlgorithmType = None):
        if Utils.results_dir.exists() is False:
            Utils.results_dir.mkdir(parents=True, exist_ok=True)

        if algorithm:
            filename = f"{algorithm.value}_{filename}"

        with open(Utils.results_dir / filename, "w") as f:
            json.dump(results, f, indent=4)

    @staticmethod
    def save_model(model: torch.nn.Module,example_input: torch.Tensor, filename="model.pth",algorithm: AlgorithmType = None):
        if Utils.results_dir.exists() is False:
            Utils.results_dir.mkdir(parents=True, exist_ok=True)

        import torch
        if algorithm:
            filename = f"{algorithm.value}_{filename}"

        torch.save(model.state_dict(), Utils.results_dir / filename)
        torch.onnx.export(
            model, 
            example_input,
            Utils.results_dir / filename.replace('.pth', '.onnx'), 
            export_params=True
        )

    @staticmethod
    def generate_frozen_lake_desc(width: int, height: int, hole_probability: float, random_seed: int = None) -> list[str]:
        """Generates a random Frozen Lake environment description."""
        if random_seed is not None:
            np.random.seed(random_seed)

        desc = []
        for h in range(height):
            row = ""
            for w in range(width):
                if (h == 0 and w == 0):
                    row += 'S'  # Start
                elif (h == height - 1 and w == width - 1):
                    row += 'G'  # Goal
                else:
                    if np.random.rand() < hole_probability:
                        row += 'H'  # Hole
                    else:
                        row += 'F'  # Frozen
            desc.append(row)
        return desc