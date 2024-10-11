import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from pone.models import OLSRegression, RidgeRegression, LassoRegression
from pone.utils import design_matrix, mse, r2



class Trainer:
    """Train model"""
    def __init__(self, degrees, x1, x2, y, n_lmbdas=0) -> None:
        self._degrees = degrees
        self._P = len(degrees)
        self._x1 = x1
        self._x2 = x2
        self._y = y.ravel()
        self._n_lmbdas = n_lmbdas

    # @property
    # def degrees(self):
    #     return self._degrees
    
    @property
    def lmbdas(self):
        return self._lmbdas
    
        
    @property
    def mse(self):
        return self._train_mse, self._test_mse
    

    @property
    def r2(self):
        return self._train_r2, self._test_r2
    
    @property
    def bs_metrics(self):
        return self._bs_metrics
    
    @property
    def cv_metrics(self):
        return self._cv_metrics
    

    # @degrees.setter
    # def degrees(self, degrees):
    #     self._degrees = degrees

    def _get_scaled(self, X, seed):
        X_train, X_test, y_train, y_test = train_test_split(X[:, 1:], self._y, test_size=0.2, random_state=seed)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = np.column_stack((np.ones(X_train.shape[0]), scaler.transform(X_train)))
        X_test_scaled = np.column_stack((np.ones(X_test.shape[0]), scaler.transform(X_test)))
        return X_train_scaled, X_test_scaled, y_train, y_test
    

    def _k_fold(self, X, k):
        """Find indices for train and test data for k fold split.
        
        Args:
            X (np.ndarray): input features
            k (int): number of folds
            
        Returns:
            np.ndarray: tuples of indices for train and test data
        """
        n_samples = len(X)
        fold_size = n_samples // k
        idx = np.arange(n_samples)
        np.random.shuffle(idx)
        k_fold_indices = []

        for i in range(k):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size
            test_indices = idx[test_start:test_end]
            train_indices = np.concatenate([idx[:test_start], idx[test_end:]])
            k_fold_indices.append((train_indices, test_indices))
            
        return k_fold_indices
    
    def run_ols(self, scale=False):
        self._train_mse = np.zeros(self._P)
        self._train_r2 = np.zeros(self._P)
        self._test_mse = np.zeros(self._P)
        self._test_r2 = np.zeros(self._P)
        self._betas = []

        for i in tqdm(range(self._P)):
            X = design_matrix(self._x1, self._x2, self._degrees[i])
            
            if scale:
                X_train, X_test, y_train, y_test = train_test_split(X[:, 1:], self._y, test_size=0.2)
                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train = np.column_stack((np.ones(X_train.shape[0]), scaler.transform(X_train)))
                X_test = np.column_stack((np.ones(X_test.shape[0]), scaler.transform(X_test)))

            else: 
                X_train, X_test, y_train, y_test = train_test_split(X, self._y, test_size=0.2)

            model = OLSRegression()
            model.fit(X_train, y_train)

            beta = model.beta
            self._betas.append(beta)
            # beta_mean.append(np.mean(beta))
            # beta_std.append(np.std(beta))

            y_tilde = model.predict(X_train)
            y_pred = model.predict(X_test)

            self._train_mse[i] = mse(y_train, y_tilde)
            self._train_r2[i] = r2(y_train, y_tilde)
            self._test_mse[i] = mse(y_test, y_pred)
            self._test_r2[i] = r2(y_test, y_pred)



    def run_ridge(self, lmbda_range, include_ols=False, scale=False):
        self._lmbdas = np.logspace(lmbda_range[0], lmbda_range[1], self._n_lmbdas) # [1/(10**i) for i in range(self._n_lmbdas-1, -1, -1)]
        if include_ols:
            self._lmbdas = np.concatenate((self._lmbdas, np.array([0])))
            self._n_lmbdas += 1
        self._train_mse = np.zeros((self._P, self._n_lmbdas))
        self._test_mse = np.zeros((self._P, self._n_lmbdas))

        for i in tqdm(range(self._P)):
            X = design_matrix(self._x1, self._x2, self._degrees[i])

            if scale:
                X_train, X_test, y_train, y_test = train_test_split(X[:, 1:], self._y, test_size=0.2)
                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train = np.column_stack((np.ones(X_train.shape[0]), scaler.transform(X_train)))
                X_test = np.column_stack((np.ones(X_test.shape[0]), scaler.transform(X_test)))

            else: 
                X_train, X_test, y_train, y_test = train_test_split(X, self._y, test_size=0.2)

            for j, lmbda in enumerate(self._lmbdas):
                model = RidgeRegression(lmbda)
                model.fit(X_train, y_train)

                y_tilde = model.predict(X_train)
                y_pred = model.predict(X_test)
                
                self._train_mse[i][j] = mse(y_train, y_tilde)
                self._test_mse[i][j] = mse(y_test, y_pred)


    def run_lasso(self, lmbda_range, include_ols=False, scale=False):
        self._lmbdas = np.logspace(lmbda_range[0], lmbda_range[1], self._n_lmbdas) # [1/(10**i) for i in range(self._n_lmbdas-1, -1, -1)]
        if include_ols:
            self._lmbdas = np.concatenate((self._lmbdas, np.array([0])))
            self._n_lmbdas += 1
        self._train_mse = np.zeros((self._P, self._n_lmbdas))
        self._test_mse = np.zeros((self._P, self._n_lmbdas))

        for i in tqdm(range(self._P)):
            X = design_matrix(self._x1, self._x2, self._degrees[i])
            
            if scale:
                X_train, X_test, y_train, y_test = train_test_split(X[:, 1:], self._y, test_size=0.2)
                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train = np.column_stack((np.ones(X_train.shape[0]), scaler.transform(X_train)))
                X_test = np.column_stack((np.ones(X_test.shape[0]), scaler.transform(X_test)))

            else: 
                X_train, X_test, y_train, y_test = train_test_split(X, self._y, test_size=0.2)
            
            for j, lmbda in enumerate(self._lmbdas):
                model = LassoRegression(lmbda)
                model.fit(X_train, y_train)

                y_tilde = model.predict(X_train)
                y_pred = model.predict(X_test)
                
                self._train_mse[i][j] = mse(y_train, y_tilde)
                self._test_mse[i][j] = mse(y_test, y_pred)


    def run_bootstrap(self, n_bootstraps, seed, scale=False):
        error_train = np.zeros(self._P)
        error_test = np.zeros(self._P)
        bias = np.zeros(self._P)
        variance = np.zeros(self._P)

        model = OLSRegression()

        for i in tqdm(range(self._P)):
            X = design_matrix(self._x1, self._x2, self._degrees[i])
            if scale:
                X_train, X_test, y_train, y_test = self._get_scaled(X, seed)

            else: 
                X_train, X_test, y_train, y_test = train_test_split(X, self._y, test_size=0.2, random_state=seed)

            y_tilde = np.empty((y_train.shape[0], n_bootstraps))
            y_pred = np.empty((y_test.shape[0], n_bootstraps))
            mse_train = np.empty(n_bootstraps)
            mse_test = np.empty(n_bootstraps)

            for j in range(n_bootstraps):
                X_, y_ = resample(X_train, y_train)
                model.fit(X_, y_)

                y_tilde[:, j] = model.predict(X_).ravel()
                y_pred[:, j] = model.predict(X_test).ravel()

                mse_train[j] = mse(y_, y_tilde[:, j])
                mse_test[j] = mse(y_test, y_pred[:, j])

            error_train[i] = np.mean(mse_train)
            error_test[i] = np.mean(mse_test)
            bias[i] = np.mean((y_test - np.mean(y_pred, axis=1))**2)
            variance[i] = np.mean(np.var(y_pred, axis=1, keepdims=True))

        self._bs_metrics = (error_train, error_test, bias, variance)


    def run_crossval(self, k, lmbda=1.0e-05, scale=False):
        error_ols_train = np.zeros(self._P)
        error_ols_test = np.zeros(self._P)
        error_ridge_train = np.zeros(self._P)
        error_ridge_test = np.zeros(self._P)
        error_lasso_train = np.zeros(self._P)
        error_lasso_test = np.zeros(self._P)
        # bias = np.zeros(self._P)
        # variance = np.zeros(self._P)

        ols = OLSRegression()
        ridge = RidgeRegression(lmbda=lmbda)
        lasso = LassoRegression(lmbda=lmbda)

        for i in tqdm(range(self._P)):
            X = design_matrix(self._x1, self._x2, self._degrees[i])

            cv_ols_train = np.zeros(k)
            cv_ols_test = np.zeros(k)
            cv_ridge_train = np.zeros(k)
            cv_ridge_test = np.zeros(k)
            cv_lasso_train = np.zeros(k)
            cv_lasso_test = np.zeros(k)

            k_fold_idx = self._k_fold(X, k)

            for j, (train_indices, test_indices) in enumerate(k_fold_idx):
                X_train, X_test = X[train_indices], X[test_indices]
                if scale: 
                    scaler = StandardScaler()
                    scaler.fit(X_train[:, 1:])
                    X_train = np.column_stack((np.ones(X_train.shape[0]), scaler.transform(X_train[:, 1:])))
                    X_test = np.column_stack((np.ones(X_test.shape[0]), scaler.transform(X_test[:, 1:])))

                y_train, y_test = self._y[train_indices], self._y[test_indices]
                # OLS
                ols.fit(X_train, y_train)
                y_tilde = ols.predict(X_train).ravel()
                y_pred = ols.predict(X_test).ravel()
                cv_ols_train[j] = mse(y_train, y_tilde)
                cv_ols_test[j] = mse(y_test, y_pred)
                # Ridge
                ridge.fit(X_train, y_train)
                y_tilde = ridge.predict(X_train).ravel()
                y_pred = ridge.predict(X_test).ravel()
                cv_ridge_train[j] = mse(y_train, y_tilde)
                cv_ridge_test[j] = mse(y_test, y_pred)
                # Lasso
                lasso.fit(X_train, y_train)
                y_tilde = lasso.predict(X_train).ravel()
                y_pred = lasso.predict(X_test).ravel()
                cv_lasso_train[j] = mse(y_train, y_tilde)
                cv_lasso_test[j] = mse(y_test, y_pred)

            error_ols_train[i] = np.mean(cv_ols_train)
            error_ols_test[i] = np.mean(cv_ols_test)
            error_ridge_train[i] = np.mean(cv_ridge_train)
            error_ridge_test[i] = np.mean(cv_ridge_test)
            error_lasso_train[i] = np.mean(cv_lasso_train)
            error_lasso_test[i] = np.mean(cv_lasso_test)
        train_metrics = (error_ols_train, error_ridge_train, error_lasso_train)
        test_metrics = (error_ols_test, error_ridge_test, error_ridge_test)
        self._cv_metrics = (train_metrics, test_metrics)


    def plot_beta(self, figname=None):
        M = len(self._betas)
        c = sns.color_palette("mako", n_colors=M, as_cmap=False)

        fig, ax = plt.subplots()
        for i in range(M):
            ax.plot(self._betas[i], 'o-', color=c[i], label=f"{i}")

        ax.legend(title="Degree", loc="upper right")
        ax.set_xticks(np.arange(len(self._betas[-1])))

        ax.set_xlabel(r"$i$")
        ax.set_ylabel(r"$\beta$-value")

        if figname is not None:
            fig.savefig(f"latex/figures/{figname}.pdf")
            
        else:
            plt.show()


    def plot_mse_r2(self, figname=None):
        c = sns.color_palette("mako", n_colors=2, as_cmap=False)

        fig, ax = plt.subplots(layout='constrained')
        ax2 = ax.twinx()
        ax.plot(self._degrees, self._test_mse, color=c[0],  label="MSE")
        ax2.plot(self._degrees, self._test_r2, color=c[1], label=r"R$^{2}$")

        ax2.grid(None)

        mse_lines, mse_labels = ax.get_legend_handles_labels()
        r2_lines, r2_labels = ax2.get_legend_handles_labels() 
        ax2.legend(mse_lines+r2_lines, mse_labels+r2_labels, loc="upper right")

        ax.set_xlabel("Polynomial degree")
        ax.set_ylabel("MSE")
        ax2.set_ylabel(r"R$^{2}$")

        if figname is not None:
            fig.savefig(f"latex/figures/{figname}.pdf", bbox_inches="tight")
        
        else:
            plt.show()


    def plot_lmbda_mse(self, figname=None, include_ols=False):
        c = sns.color_palette("mako", n_colors=self._n_lmbdas, as_cmap=False)

        fig, ax = plt.subplots(layout='constrained')

        if include_ols:
            for i in range(self._n_lmbdas-1):
                ax.plot(self._degrees, self._test_mse.T[i], color=c[i], label=rf"$\lambda_{i+1}$")
            ax.plot(self._degrees, self._test_mse.T[-1], "--", color=c[-1], label="OLS")
        else:
            for i in range(self._n_lmbdas):
                ax.plot(self._degrees, self._test_mse.T[i], color=c[i], label=rf"$\lambda_{i+1}$")

        ax.legend(loc="upper right")
        ax.set_yscale("log")
        ax.set_xlabel("Polynomial degree")
        ax.set_ylabel("MSE")


        if figname is not None:
            fig.savefig(f"latex/figures/{figname}.pdf", bbox_inches="tight")
            
        else:
            plt.show()


    def plot_bootstrap(self, figname=None):
        n_colors = len(self._bs_metrics)
        c = sns.color_palette("mako", n_colors=n_colors, as_cmap=False)

        fig, ax = plt.subplots(layout='constrained')
        ax.plot(self._degrees, self._bs_metrics[0], color=c[0], label="Train")
        ax.plot(self._degrees, self._bs_metrics[1], color=c[1], label="Test")
        ax.plot(self._degrees, self._bs_metrics[2], color=c[2], label="Bias")
        ax.plot(self._degrees, self._bs_metrics[3], color=c[3], label="Variance")

        ax.legend(loc="upper right")
        # ax.set_yscale("log")
        ax.set_xlabel("Polynomial degree")
        ax.set_ylabel("MSE")


        if figname is not None:
            fig.savefig(f"latex/figures/{figname}.pdf", bbox_inches="tight")
            
        else:
            plt.show()


    def plot_crossval(self, figname=None):
        train_metrics = self._cv_metrics[0]
        test_metrics = self._cv_metrics[1]
        n_colors = len(train_metrics)
        c = sns.color_palette("mako", n_colors=n_colors, as_cmap=False)

        fig, ax = plt.subplots(layout='constrained')
        # Test
        ax.plot(self._degrees, test_metrics[0], color=c[0], label="OLS")
        ax.plot(self._degrees, test_metrics[1], color=c[1], label="Ridge")
        ax.plot(self._degrees, test_metrics[2], color=c[2], label="Lasso")
        # Train
        ax.plot(self._degrees, train_metrics[0], "--", color=c[0])
        ax.plot(self._degrees, train_metrics[1], "--", color=c[1])
        ax.plot(self._degrees, train_metrics[2], "--", color=c[2])

        ax.legend(loc="upper right")
        # ax.set_yscale("log")
        ax.set_xlabel("Polynomial degree")
        ax.set_ylabel("MSE")


        if figname is not None:
            fig.savefig(f"latex/figures/{figname}.pdf", bbox_inches="tight")
            
        else:
            plt.show()


    def plot_bs_cv(self, figname=None):
        train_metrics = self._cv_metrics[0]
        test_metrics = self._cv_metrics[1]
        n_colors = len(train_metrics) + 1
        c = sns.color_palette("mako", n_colors=n_colors, as_cmap=False)

        fig, ax = plt.subplots(layout='constrained')
        # CV
        ax.plot(self._degrees, test_metrics[0], color=c[0], label="CV OLS")
        ax.plot(self._degrees, test_metrics[1], color=c[1], label="CV Ridge")
        ax.plot(self._degrees, test_metrics[2], color=c[2], label="CV Lasso")
        ax.plot(self._degrees, self._bs_metrics[1], "--", color=c[3], label="Bootstrap")

        ax.legend(loc="upper right")
        # ax.set_yscale("log")
        ax.set_xlabel("Polynomial degree")
        ax.set_ylabel("MSE")


        if figname is not None:
            fig.savefig(f"latex/figures/{figname}.pdf", bbox_inches="tight")
            
        else:
            plt.show()


    def plot_mse(self, figname=None, include_train=False):
        c = sns.color_palette("mako", n_colors=2, as_cmap=False)

        fig, ax = plt.subplots(layout='constrained')

        ax.plot(self._degrees, self._test_mse, color=c[0],  label="MSE")
        
        if include_train:
            ax.plot(self._degrees, self._train_mse, "--", color=c[0], label=r"MSE$_{train}$")

        ax.legend(loc="upper right")

        ax.set_xlabel("Polynomial degree")
        ax.set_ylabel("MSE")

        if figname is not None:
            fig.savefig(f"latex/figures/{figname}.pdf", bbox_inches="tight")
            
        else:
            plt.show()

