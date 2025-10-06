## project: setup 🏗️
- `dataset` should contain the training data `food_truck.txt` and `housing_prices.txt`
- `lib` contains the the custom implementations of linear regression
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
```

## improvements for future dev 💻
- `code structure & Readability`:
  - while it's nice to have a gradient descent and cost function they are very regid so implementing a other cost function can be generalized by adding a layer of abstraction.
    furthermore we can also add a predict function instead of calculating it inside the plot. A better alternative is like scikit's API in which we create a base class that contains
    all relevant functions and then implement our own linear regression on top of it
  - add Typing for better code readability and code completion
  - add Docstrings for some explanation of what a function actually does 
- `Stability & Performance`
  - I've implemented my "own" StdScalar but that's not enough gradient descent can AND WILL overflow (or underflow) so we have to scale the data appropriatly
  - I use `y = y.to_numpy().reshape(-1, 1)` but also `y.ravel()`. which makes not very robust to keep it consistend
  - Stopping criteria should be added (epsilon) such that we can allow the Linear Regression to stop after the error is in a acceptable margin. Right now it just runs for n iterations
- `Extensibility`
  - seperate plotting from the Linear regression class (and take snapshots) as a utils package/module instead
  - add a Example module
  - like scikit's LinearRegression add a fit_intercept option and a other cost function perhaps (instead of MSE) like MAE , RMSE, LSE, R² to conform to multiple use cases
  
