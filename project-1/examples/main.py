import numpy as np
import warnings

from pone.utils import set_plt_params
from pone.data_generation import create_function_data, create_terrain_data
from pone.trainer import Trainer
from pone.plot import plot_terrain, plot_surf, plot_heatmap



def part_a(degrees, N):
    print("------------ Running part a ------------")
    print("Franke function without noise:")
    x1, x2, y = create_function_data(N=N)
    
    trainer = Trainer(degrees, x1, x2, y)
    trainer.run_ols()
    trainer.plot_beta(figname=f"ols_beta_smooth_N{N}")
    trainer.plot_mse_r2(figname=f"ols_error_smooth_N{N}")

    print("Scaled:")
    trainer = Trainer(degrees, x1, x2, y)
    trainer.run_ols(True)
    trainer.plot_beta(figname=f"ols_beta_smooth_scaled_N{N}")
    trainer.plot_mse_r2(figname=f"ols_error_smooth_scaled_N{N}") 
    
    print("Franke function with noise:")
    x1, x2, y = create_function_data(N=N, noise_var=0.1)

    trainer = Trainer(degrees, x1, x2, y)
    trainer.run_ols()
    trainer.plot_beta(figname=f"ols_beta_N{N}")
    trainer.plot_mse_r2(figname=f"ols_error_N{N}")

    print("Scaled:")
    trainer = Trainer(degrees, x1, x2, y)
    trainer.run_ols(True)
    trainer.plot_beta(figname=f"ols_beta_scaled_N{N}")
    trainer.plot_mse_r2(figname=f"ols_error_scaled_N{N}")
    


def part_b(degrees, N, n_lmbdas):
    print("------------ Running part b ------------")
    x1, x2, y = create_function_data(N=N, noise_var=0.1)
    lmbda_range = [-5, -2]

    trainer = Trainer(degrees, x1, x2, y, n_lmbdas)
    trainer.run_ridge(lmbda_range, include_ols=True)
    trainer.plot_lmbda_mse(f"ridge_error_N{N}", include_ols=True)

    print("Scaled:")
    trainer = Trainer(degrees, x1, x2, y, n_lmbdas)
    trainer.run_ridge(lmbda_range, include_ols=True, scale=True)
    trainer.plot_lmbda_mse(f"ridge_error_scaled_N{N}", include_ols=True)

    print(f"Lambda values: {trainer.lmbdas}")


def part_c(degrees, N, n_lmbdas):
    print("------------ Running part c ------------")
    x1, x2, y = create_function_data(N=N, noise_var=0.1)
    lmbda_range = [-5, -2]

    trainer = Trainer(degrees, x1, x2, y, n_lmbdas)
    trainer.run_lasso(lmbda_range, include_ols=True)
    trainer.plot_lmbda_mse(f"lasso_error_N{N}", include_ols=True)

    print("Scaled:")
    trainer = Trainer(degrees, x1, x2, y, n_lmbdas)
    trainer.run_lasso(lmbda_range, include_ols=True, scale=True)
    trainer.plot_lmbda_mse(f"lasso_error_scaled_N{N}", include_ols=True)

    print(f"Lambda values: {trainer.lmbdas}")
    

# def part_d():
#     pen and paper


def part_e(degrees, N, n_bootstraps, seed):
    print("------------ Running part e ------------")
    x1, x2, y = create_function_data(N=N, noise_var=1)

    trainer = Trainer(degrees, x1, x2, y)
    trainer.run_ols()
    trainer.plot_mse(f"ols_hastie_N{N}", include_train=True)

    x1, x2, y = create_function_data(N=N, noise_var=0.1)

    trainer = Trainer(degrees, x1, x2, y)
    trainer.run_bootstrap(n_bootstraps, seed)
    trainer.plot_bootstrap(f"ols_bootstrap_{n_bootstraps}_N{N}")

    print("Scaled:")
    trainer = Trainer(degrees, x1, x2, y)
    trainer.run_bootstrap(n_bootstraps, seed, scale=True)
    trainer.plot_bootstrap(f"ols_bootstrap_{n_bootstraps}_scaled_N{N}")


def part_f(degrees, N, k, n_bootstraps):
    print("------------ Running part f ------------")
    x1, x2, y = create_function_data(N=N, noise_var=0.1)

    trainer = Trainer(degrees, x1, x2, y)
    trainer.run_crossval(k)
    trainer.plot_crossval(f"ols_crossval_{k}_N{N}")

    trainer.run_bootstrap(n_bootstraps, seed)
    trainer.plot_bs_cv(f"ols_bs_crossval_{k}_N{N}")


def part_g(degrees, N, n_lmbdas, n_bootstraps, seed, k):
    print("------------ Running part g ------------")
    terrain = "../data/n36_e110_1arc_v3.tif"
    lmbda_range = [-5, -2]
    x1, x2, y = create_terrain_data(terrain, N, scaled=True)
    plot_surf(x1, x2, y, f"terrain_N{N}")

    print("OLS beta:")
    d = np.arange(1, 6)
    trainer = Trainer(d, x1, x2, y)
    trainer.run_ols(scale=True)
    trainer.plot_beta(figname=f"ols_beta_terrain_D5_N{N}")
    trainer.plot_mse_r2(figname=f"ols_error_terrain_D5_N{N}")

    print("OLS scaled:")
    trainer = Trainer(degrees, x1, x2, y)
    trainer.run_ols(scale=True)
    trainer.plot_mse_r2(figname=f"ols_error_terrain_N{N}") 

    print("Ridge scaled:")
    trainer = Trainer(degrees, x1, x2, y, n_lmbdas)
    trainer.run_ridge(lmbda_range, include_ols=True, scale=True)
    trainer.plot_lmbda_mse(f"ridge_error_terrain_N{N}", include_ols=True)

    print("Lasso scaled:")
    trainer = Trainer(degrees, x1, x2, y, n_lmbdas)
    trainer.run_lasso(lmbda_range, include_ols=True, scale=True)
    trainer.plot_lmbda_mse(f"lasso_error_terrain_N{N}", include_ols=True)

    print("Hastie figure:")
    trainer = Trainer(degrees, x1, x2, y)
    trainer.run_ols(scale=True)
    trainer.plot_mse(include_train=True, figname=f"ols_hastie_terrain_N{N}")

    print("Bootstrap:")
    trainer = Trainer(degrees, x1, x2, y)
    trainer.run_bootstrap(n_bootstraps, seed, scale=True)
    trainer.plot_bootstrap(f"ols_bootstrap_{n_bootstraps}_terrain_N{N}")

    print("Cross-validation")
    trainer.run_crossval(k)
    trainer.plot_crossval(f"ols_crossval_{k}_terrain_N{N}")

    print("Bootstrap and cross-validation")
    trainer.plot_bs_cv(f"ols_bs_crossval_{n_bootstraps}_{k}_terrain_N{N}")



def main_auto(seed):
    # Prettify plots
    set_plt_params()

    # Set parameters to use for part a
    degrees = np.arange(1, 6)
    N = 50
    part_a(degrees, N)

    # Set parameters to use for the rest of the project
    degrees = np.arange(1, 15)
    N = 50
    n_lmbdas = 8
    part_b(degrees, N, n_lmbdas)
    part_c(degrees, N, n_lmbdas)

    N = 50
    n_bootstraps = 100
    degrees = np.arange(1, 18)
    part_e(degrees, N, n_bootstraps, seed)

    N = 50 
    k = 10
    n_bootstraps = 100
    degrees = np.arange(1, 18)
    part_f(degrees, N, k, n_bootstraps)
    
    N = 50
    n_lmbdas = 8
    k = 10
    n_bootstraps = 100
    degrees = np.arange(1, 15)
    part_g(degrees, N, n_lmbdas, n_bootstraps, seed, k)


def main_input():
    # N = int(input("Define number of data points: "))
    # P = int(input("Define polynomial degree: "))
    print("Not implemented yet!")
    

if __name__ == '__main__':
    # Filter out warnings of lasso not converging
    warnings.filterwarnings('ignore')

    auto = input("Run program with pre-set parameters? [y/n] ")

    if auto == "n":
        main_input()
    else:
        seed = 2024
        np.random.seed(seed)
        print(f"Auto with seed: {seed}\n")
        main_auto(seed)
    