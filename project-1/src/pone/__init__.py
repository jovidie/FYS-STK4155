#!/usr/bin/env python
# -*- coding: utf-8 -*-


import importlib.metadata

from .models import OLSRegression, RidgeRegression, LassoRegression
from .utils import design_matrix, franke_function, set_plt_params, mse, r2
from .data_generation import create_function_data, create_terrain_data
from .trainer import Trainer
from .plot import plot_surf, plot_terrain, plot_heatmap

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
    "create_terrain_data",
    "Trainer", 
    "plot_surf", 
    "plot_terrain",
    "plot_heatmap"
]
