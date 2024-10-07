# FYS-STK4155 Project 1

In this project I have studied regression methods, specifically the Ordinary Least Squares (OLS), Ridge and Lasso, and analyzed their performance on synthetic and real data.


## Set up environment
The `environment.yml` file contains all packages necessary to build and work with the `pone` package. Install and activate the `conda` environment using the following command:
```sh
conda env create --file environment.yml
conda activate pone-dev
```

To update an existing environment:
```sh
conda env update --name pone-dev --file environment.yml --prune
```

The dependencies can also be installed directly from `requirements.txt`:
```sh
python3 -m pip install -r requirements.txt
```


## Installation
To install this project, run the following command:
```sh
python3 -m pip install -e .
```


## Project 1 structure
```sh
.
├── latex
│   ├── figures/
│   ├── sections/
│   ├── main.tex
│   ├── main.pdf
│   └── references.bib
├── notebooks
│   ├── exploration.ipynb
│   └── project.ipynb
├── src
│   └── pone
│       ├── __init__.py
│       ├── data_generation.py
│       ├── models.py
│       ├── resamplers.py
│       └── utils.py
├── tests
│   └── test.py
├── environment.yml  
├── pyproject.toml
├── README.md
└── requirements.txt
```
