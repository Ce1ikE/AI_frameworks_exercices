## project: setup 🏗️
- `dataset` contains the country dataset
- `lib` contains the the plotting and data loading functions of this exercice
- `main.py` entrypoint

## project: how to run ? :accessibility:
First you must install the required libraries either using the `requirements.txt` or the `pyproject.toml`
```sh
# create a virtual environment either with venv or conda
python -m venv ./venv
# activate the virtual environment
./venv/scripts/activate
# and use pip
pip install -r requirements.txt

# or with uv
uv venv
uv add -r requirements.txt
```

after you can just modify and run the `main.py` , which is the entrypoint of this application
```sh
py ./main.py
# or with uv
uv run ./main.py
# or with streamlit (for the browser)
streamlit run ./main.py
```