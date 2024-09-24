# FYS-STK4155 Project 1

## project structure
```bash
.
├── latex
│   ├── figures/
│   ├── sections/
│   ├── main.tex
│   └── references.bib
├── notebooks
│   ├── exploration.ipynb
├── src
│   └── pone
│       ├── __init__.py
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

## Set up environment

The `environment.yml` file contains all packages to build and work with the `src` package. Install and activate the `conda` environment:

    $ conda env create --file environment.yml
    $ conda activate pone-dev

To update an existing environment:

    $ conda env update --name pone-dev --file environment.yml --prune

The dependencies can also be installed independent of `conda` by using `pip`:

    $ python3 -m pip install -r requirements.txt

### Install project package

From the `FYS-STK4155` repository, install the `pone` package while in the `project-1` directory:

    <!-- $ python3 -m pip install -e .
