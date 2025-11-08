from lib.exercices import *


def main():
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = RESULTS / f"gradient_descent_{timestamp}" 
    results_dir.mkdir(parents=True,exist_ok=True)
    gradient_descent_custom(results_dir)
    gradient_descent_pytorch(results_dir)

    results_dir = RESULTS / f"autograd_{timestamp}" 
    results_dir.mkdir(parents=True,exist_ok=True)
    autograd(results_dir)

if __name__ == "__main__":
    main()
