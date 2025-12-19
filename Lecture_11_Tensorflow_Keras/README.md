## project: setup 🏗️
- `datasets` just a temp folder in which pytorch will dput the downloaded dataset, tensorflow will download it when the Docker container starts 
- `lib` contains the the exercice functions and models
- `main.py` entrypoint for the pytorch exercice
- `docker-compose.tensorflow.yaml` entrypoint for the tensorflow exerice

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
for the PyTorch exerice
```sh
py ./main.py
# or with uv
uv run ./main.py
```
for the tensorflow exerice install Docker (Docker desktop if on windows)
and run the docker compose file
```sh
# once for startup
docker-compose -f docker-compose.tensorflow.yaml up --build
# or when you just modified the code
# (a volume is mapped to the code so no need to rebuild the image)
docker-compose -f docker-compose.tensorflow.yaml up
```

