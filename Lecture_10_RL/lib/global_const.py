from pathlib import Path

RESULTS_PATH = Path(__file__).parent.parent / "results"
RESULTS_PATH.mkdir(exist_ok=True)
RANDOM_SEED = 42
FONTDICT = {
    "fontsize": 14,
    "fontweight": "bold",
    "fontfamily": "monospace",
    "color": "darkblue"
}