from lib.pytorch_exercice import pytorch_example
from pathlib import Path

RESULTS = Path("./results")  
RESULTS.mkdir(parents=True, exist_ok=True)

def main():
    pytorch_example(results_path=RESULTS)

if __name__ == "__main__":
    main()
