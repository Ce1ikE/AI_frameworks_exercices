from pathlib import Path

RANDOM_SEED = 42
RESULTS = Path("results")
RESULTS.mkdir(parents=True,exist_ok=True)
DATASETS = Path("datasets")
FONTDICT = {
    "fontsize": 10,
    "fontweight": "bold",
    "fontfamily": "monospace",
}
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True,exist_ok=True)