#!/usr/bin/env python
# -*- coding: utf-8 -*-


import importlib.metadata

from .models import LinearRegression, RidgeRegression, LassoRegression
from .utils import design_matrix, franke_function, set_plt_params, mse, r2
from .resamplers import Bootstrapper

__version__ = importlib.metadata.version(__package__)

__all__ = [
    "LinearRegression",
    "RidgeRegression",
    "LassoRegression",
    "design_matrix",
    "franke_function",
    "set_plt_params",
    "mse",
    "r2",
    "Bootstrapper"
]
