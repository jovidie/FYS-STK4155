from turtle import color
import numpy as np
import seaborn as sns
from imageio.v3 import imread
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def plot_terrain(filename, country=None, figname=None):
    """Plot function or terrain data.
    
    Args:
        filename (str): read data from file
        country (str): location of the terrain data
		figname (str): saves figure with the given name 
		
	Returns:
        None
    """
    terrain = imread(filename)
    fig, ax = plt.subplots()
    ax.set_title(country)
    ax.imshow(terrain, cmap='gray')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    
    if figname is not None:
        fig.savefig(f"latex/figures/{figname}.pdf")

    else:
        plt.show()


def plot_surf(x1, x2, y, figname=None):
    """Plot function or terrain data to 3D surface.
    
    Args:
        x1 (np.ndarray): x1-values
        x2 (np.ndarray): x2-values
        y (np.ndarray): function or terrain data values
		figname (str): saves figure with the given name 
		
	Returns:
        None
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    c = sns.color_palette("mako", as_cmap=True)
    #c = sns.diverging_palette(145, 300, s=60, as_cmap=True)
    surf = ax.plot_surface(x1, x2, y, cmap=c, linewidth=0, antialiased=False)

    # Customize the z axis.
    #ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    if figname is not None:
        fig.savefig(f"latex/figures/{figname}.pdf")

    else:
        plt.show()


def plot_heatmap(P, lmbdas, mse, figname=None):
    """Plot MSE in a heatmap as a function of lambda and polynomial degree.
    
    Args:
        P (int): polynomial degree
        lmbdas (np.ndarray): lambda values
        mse (np.ndarray): MSE values
		figname (str): saves figure with the given name 
		
	Returns:
        None
    """
    degrees = np.arange(P)
    lmbdas = np.log10(lmbdas)
    lmbdas_, p_ = np.meshgrid(lmbdas, degrees)

    idx = np.where(mse == mse.min())

    fig, ax = plt.subplots()
    c = sns.color_palette("mako", as_cmap=True)
    cs = ax.contourf(lmbdas_, p_, mse, levels=len(lmbdas), cmap=c)

    # Include point where optimal parameters are
    ax.plot(lmbdas[idx[1]], degrees[idx[0]], "X", label=r"$\lambda_{{opt}}$")
    ax.legend(title=f"MSE = {mse.min():.4f}")

    fig.colorbar(cs, label="MSE")

    ax.set_xlabel(r"$Log_{10}(\lambda)$")
    ax.set_ylabel("Polynomial degree")

    if figname is not None:
        fig.savefig(f"latex/figures/{figname}.pdf", bbox_inches="tight")
        
    else:
        plt.show()