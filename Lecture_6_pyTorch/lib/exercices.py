import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import datetime as dt
import pprint
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split

from .Models import CustomModel, PyTorchModel, Model

RANDOM_SEED = 42
RESULTS = Path("results")
RESULTS.mkdir(parents=True,exist_ok=True)
SAMPLES = 100
FUNC_TO_LEARN = lambda x: np.sin(x*np.pi + 2) + 2*x + 6
STDSCALAR = lambda x,mean,std: (x - mean)/std
lr_params = [1.0,0.5,0.1,0.05,0.01,0.001]
fontdict = {
    "fontsize": 10,
    "fontweight": "bold",
    "fontfamily": "monospace",
}

def _plot_cost_and_predictions(
    model_type: CustomModel | PyTorchModel,
    X_TRAIN,
    X_TEST,
    Y_TRAIN,
    Y_TEST,
    results_dir: Path
):
    # --- plot 1: Cost over epochs  --- #
    plt.figure(figsize=(10, 6))
    plt.title(f"Cost vs Epochs for Different Learning Rates ({model_type.__name__})",fontdict=fontdict)    
    for lr in lr_params:
        model: Model = model_type(learning_rate=lr, epochs=50)
        results_training = model.train(X_TRAIN, Y_TRAIN)
        plt.plot(
            results_training["epoch"],
            results_training["cost"],
            label=f"lr={lr}"
        )

    plt.yscale("log")
    plt.xlabel("Epochs",fontdict=fontdict)
    plt.ylabel("Cost (log scale)",fontdict=fontdict)
    plt.grid(True, which="both", alpha=0.2)
    plt.legend()
    plt.savefig(results_dir / f"cost_vs_epochs_{model_type.__name__}.svg", format="svg")
    plt.close()
    
    
    # --- plot 2: true vs pred  --- #
    plt.figure(figsize=(8, 6))
    plt.title(f"Predictions vs True Values (final epoch) ({model_type.__name__})",fontdict=fontdict)
    for lr in lr_params:
        model: Model = model_type(learning_rate=lr, epochs=50)
        model.train(X_TRAIN, Y_TRAIN)
        y_pred = model.predict(X_TEST)
        plt.scatter(X_TEST, y_pred, label=f"lr={lr}", alpha=0.6)

    plt.plot(X_TEST, Y_TEST, label="True Values",linestyle=":", color="black", marker="x")
    plt.xlabel("X (normalized)",fontdict=fontdict)
    plt.ylabel("Y",fontdict=fontdict)
    plt.grid(True, which="both", alpha=0.2)
    plt.legend()
    plt.savefig(results_dir / f"predictions_vs_true_values_{model_type.__name__}.svg", format="svg")
    plt.close()




def gradient_descent_custom():    
    # ------------ DATA GATHERING ------------ #
    X = np.array([i for i in range(SAMPLES)], dtype=np.float32)
    Y = np.array([FUNC_TO_LEARN(i) for i in range(SAMPLES)], dtype=np.float32)
    # ------------ DATA CLEANING ------------ #
    # (same as custom stdScalar in Lecture 1 and 2)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X,Y,test_size=0.2,random_state=RANDOM_SEED)
    X_mean = np.mean(X_TRAIN)
    X_std = np.std(X_TRAIN)
    X_TRAIN, X_TEST = STDSCALAR(X_TRAIN,X_mean,X_std), STDSCALAR(X_TEST,X_mean,X_std)
    # ------------ PLOTTING ------------ #
    _plot_cost_and_predictions(
        CustomModel,
        X_TRAIN, X_TEST, Y_TRAIN, Y_TEST
    )

def gradient_descent_pytorch():
    # ------------ DATA GATHERING ------------ #
    X = torch.tensor([[i] for i in range(SAMPLES)], dtype=torch.float32)
    Y = torch.tensor([[FUNC_TO_LEARN(i)] for i in range(SAMPLES)], dtype=torch.float32)
    # ------------ DATA CLEANING ------------ #
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X,Y,test_size=0.2,random_state=RANDOM_SEED)
    X_mean =  X_TRAIN.mean(axis=0, keepdims=True)
    X_std = X_TRAIN.std(axis=0, keepdims=True)
    X_TRAIN, X_TEST = STDSCALAR(X_TRAIN,X_mean,X_std), STDSCALAR(X_TEST,X_mean,X_std)
    # ------------ PLOTTING ------------ #
    _plot_cost_and_predictions(
        PyTorchModel,
        X_TRAIN, X_TEST, Y_TRAIN, Y_TEST
    )


def autograd(results_dir):
    ### Exercise: 
    #   Create a function that calculates the 
    #   exp(2*x^2*sin(x)) over the range [-2, 2], with steps=25
    #   Use intermediate variables for each mathematical operation.
    
    ### Exercise: 
    #   Call the appropriate function on the variable to calculate the gradients.

    # x
    t = torch.linspace(-10,10,100,requires_grad=True)
    # sin(x)
    sin_t = t.sin()
    # x*x == x^2
    t_squared = t*t
    # x^2*sin(x)
    t_squared_times_sin_t = t_squared * sin_t
    # 2*x^2*sin(x)
    times_2_t_squared_times_sin_t = 2 * t_squared_times_sin_t
    # exp(2*x^2*sin(x))
    y = times_2_t_squared_times_sin_t.exp()
    # dy/dt 
    y.sum().backward()

    t_np = t.detach().numpy()
    y_np = y.detach().numpy()
    grad_np = t.grad.numpy()
    plt.figure(figsize=(10, 8))
    
    # --- plot 1: OG function (y) ---
    plt.subplot(2, 1, 1) 
    plt.title(r"Function $y = \exp(2x^2\sin(x))$", fontdict=fontdict)
    plt.xlabel("x",fontdict=fontdict)
    plt.ylabel("y (Log Scale)",fontdict=fontdict)
    plt.yscale("log")
    plt.plot(t_np, y_np, 'b-', label="Function y")
    plt.scatter(t_np, y_np, color='blue', marker='o', s=15)
    plt.grid(True, which="both", ls="--")
    plt.legend()
    # --- plot 2: Gradient (dy/dt) aka derivative ---
    plt.subplot(2, 1, 2)
    plt.title(r"Gradient $\frac{dy}{dt}$",fontdict=fontdict)
    plt.xlabel("x")
    plt.ylabel("dy/dt (Log Scale)",fontdict=fontdict)
    plt.yscale("log",fontdict=fontdict) 
    plt.plot(t_np, grad_np, 'r-', label="Gradient dy/dt")
    plt.scatter(t_np, grad_np, color='red', marker='x', s=15)
    plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
    plt.grid(True, which="both", ls="--")
    plt.legend()

    plt.tight_layout()
    plt.savefig(results_dir / f"function_and_gradients.svg", format="svg")
    plt.close()

    ## When exaclty you need and don't need Autograd?
    # - so autograd enables us to do auto differention (so the backwards pass in a neural net e.g.)
    #   it does this by retaining a acyclic that has recorded all the operations that created the data
    #   x -> x*2 -> x*2 + 5 -> (x*2 + 5)/6 -> etc...
    #   so as i mentioned it retains all operations to perform the backward pass when i want to update my params
    #   so Autograd is usefull during "training" of my NN but during the inferenence much less 
    #   as we just want to perform operations with the trained weights and don't want to update our weights 