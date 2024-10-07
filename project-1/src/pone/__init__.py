#!/usr/bin/env python
# -*- coding: utf-8 -*-


import importlib.metadata

from .models import OLSRegression, RidgeRegression, LassoRegression
from .utils import design_matrix, franke_function, set_plt_params, mse, r2
from .resamplers import Resampler
from .data_generation import create_function_data, create_terrain_data

__version__ = importlib.metadata.version(__package__)

__all__ = [
    "OLSRegression",
    "RidgeRegression",
    "LassoRegression",
    "design_matrix",
    "franke_function",
    "set_plt_params",
    "mse",
    "r2",
    "Resampler",
    "create_function_data",
    "create_terrain_data"
]
