## project: setup
- `dataset` should contain the training data `food_truck.txt` and `housing_prices.txt`
- `lib` contains the the custom implementations of linear regression

## project: how to run ?
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

## improvements 
- seperate plotting from the Linear regression class as a utils package/module instead
- add a Example module
- like scikit's LinearRegression add a fit_intercept option and a other cost function perhaps (instead of MSE) like MAE , RMSE, LSE, R² to conform to multiple use cases
- abstract the  LinearRegressionCustom class such that different classes can be build on top of it 
