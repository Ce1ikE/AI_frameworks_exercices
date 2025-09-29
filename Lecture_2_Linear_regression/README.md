## project: setup
- `dataset/input` should contain the training data `food_truck.txt` and `housing_prices.txt`
- `lib` contains the the custom implementations of linear regression

## project: how to run ?
First you must install the required libraries either using the `requirements.txt` or the `pyproject.toml`
```sh
# create a virtual environment either with venv
python -m venv ./venv
# activate the virtual environment
./venv/scripts/activate
# and use pip
pip install -r requirements.txt

# or with uv
uv add -r requirements.txt
```

after you can just modify and run the `main.py` , which is the entrypoint of this application

```sh
# -v for verbose output
python main.py -v 
```
