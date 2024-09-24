# FYS-STK4155 Coursework

### Set up development environment

The `environment.yml` file contains all packages to build and work with the `src` package.

Install the `conda` environment:

    $ conda env create --file environment.yml

Activate the `conda` environment:

    $ conda activate fys-stk4155

If the contents of the `environment.yml` are updated, update an existing environment with:

    $ conda env update --name fys-stk4155 --file environment.yml --prune

Optionally, the dependencies can be installed independent of `conda` by using `pip` directly:

    $ python3 -m pip install -r requirements.txt

### Install the project package

`cd` into the root of the repository and install the `src` package in editable mode from source:

    <!-- $ python3 -m pip install -e .

### pre-commit
We use [pre-commit](https://pre-commit.com/) to run Git hooks on every commit to identify simple issues such as trailing whitespace or not complying with the required formatting. Our pre-commit configuration is specified in the `.pre-commit-config.yml` file.

To set up the Git hook scripts specified in `.pre-commit-config.yml`, run

    $ pre-commit install

> **NOTE:**  If `pre-commit` identifies formatting issues in the commited code, the pre-commit Git hooks will reformat the code. If code is reformatted, it will show up in your unstaged changes. Stage them and recommit to successfully commit your changes.

It is also possible to run the pre-commit hooks without attempting a commit:

    $ pre-commit run --all-files -->