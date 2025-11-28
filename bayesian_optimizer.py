import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from matplotlib import gridspec

class BayesianOptimizer:
    """
    This is a Bayesian Optimizer that has the function to to optimize and to find the best learning_rate hyperparameter within a bounded search. The acquisition function is WEI (weighted expected improvement).

    Args:
        f: function
            Function to optimize
            
        gp: GaussianProcessRegressor
            Gaussian Process Regressor (used for regression)

        mode: str
            "lin" or "log"
            
        bound: list
            List containing upper and lower bound of the search space
            
        size_search_space: int
            Number of evaluation points used for finding the maximum of the acquisition function. Can be interpreted as the size of the discrete search space.

        search_space: ndarray
            Vector covering the search space

        gp_search_space: ndarray
            Search space of the gp can be transformed logarithmically depending on the mode.

        
        dataset: list
            The given dataset.

        states: list
            List containing the state of each iteration in the optimizing process.
    """

    def __init__(self, f, gp, mode, bound, path, size_search_space=250):
        self.f = f
        self.gp = gp
        self.path = path

        if mode not in ["lin", "log"]:
            raise ValueError(f"{mode} not supported. Please choose lin (linear) or log (logarithmic) as a mode.")
        else:
            self.mode = mode
        
        self.min = bound[0]
        self.max = bound[1]
        self.size_search_space = size_search_space
        
        if mode == "lin":
            self.search_space = np.linspace(self.min, self.max, num=size_search_space).reshape(-1, 1)
            self.gp_search_space = self.search_space
        else:
            if self.min <= 0:
                raise ValueError(f"Lower bound must be > 0 for log-mode, got {self.min}")
            log_min = np.log10(self.min)
            log_max = np.log10(self.max)
            self.search_space = np.logspace(log_min, log_max, num=size_search_space).reshape(-1, 1)
            self.gp_search_space = np.log10(self.search_space)
            
        self.dataset = []
        self.states = []

    # How to choose alpha? It's also in the paper...
    # probably use the SAWEI method. It's probably too much for this task.
    # so if alpha=0.5 it's a standard EI
    def _wei(self, f_best, alpha=0.5):
        """
        Weighted Expected Improvement (WEI) acquisition function.
        Formula after: Self-Adjusting Weighted Expected Improvement for Bayesian Optimization (Benjamins et al. 2023)

        Args:
            f_best: float
                best function value

            alpha: float in [0, 1]
                weight für exploitation (alpha) vs. exploration (1-alpha)

        Returns:
            wei: ndarray
                WEI values for the whole search space
        """

        mu, sigma = self.gp.predict(self.gp_search_space, return_std=True)

        # if sigma is zero use a very small value
        sigma = np.maximum(sigma, 1e-12)

        # turnaround mu and f_best for maximum problem
        # source: Brochu et al. (2024): A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning
        z = (mu - f_best) / sigma

        exploitation = z * sigma * norm.cdf(z) # cumulated normal distribution
        exploration = sigma * norm.pdf(z) # normal distribution

        wei = alpha * exploitation + (1.0 - alpha) * exploration

        return wei

    
    def _max_acq(self):
        """
        Calculates the next imcumbent for the current Dataset D


        Returns
            x_max: float
                Location (x-coordinate) of the best incumbent.

            util_max: float
                Utility of the best incumbent

            util: ndarray
                Utility function of the search space.
        """
        # get the value of the best incumbent
        # loss is made negative. so we want the x coordinate for the hightest loss value.
        c_inc = np.max(np.array(self.dataset)[:, 1])

        # calculate the utility function
        util = self._wei(c_inc)

        # check if utilization all zero
        # not used because of sobol
        if np.all(util == 0.):
            print("Warning! Utilization function is all zero. Returning a random point for evaluation!")
            x_max = self.search_space.reshape(-1)[np.random.randint(len(self.search_space))]
            util_max = 0

        else:
            # get the maximum's location and utility
            idx = util.argmax()
            x_max = self.search_space.reshape(-1)[idx]
            util_max = util[idx]

        return x_max, util_max, util

    
    def eval(self, n_iter=3, init_x_max=None):
        """
        Runs n_iter iteration and optimizes the function's parameters using Bayesian Optimization.

        Args:
            n_iter: int
                Iterations
            init_x_max: float
                Initial guess of the parameter. If none, a random initial guess is sampled in the search space


        Returns:
            best_return_x: float
                Best sample found during iterations.
            best_return_param:
                Parameters defining the best functions.
        """

        if not init_x_max:
            x_max = self.search_space[np.random.randint(len(self.search_space))].item()
        else:
            x_max = init_x_max

        
        # for storing the best return and some parameters specifying it
        best_return = None
        best_return_x = None
        best_return_param = None

        iterations = len(self.dataset)

        for i in range(n_iter):

            print(f"BO Iteration {iterations+i+1} --> Chosen parameter: {x_max}")

            y, param = self.f(x_max)

            # store if it's the best
            if (best_return is None) or (y > best_return):
                best_return = y
                best_return_x = x_max
                best_return_param = param
            
            self.dataset.append([x_max, y])
                
            # get all the data samples in the dataset
            xs = np.array(self.dataset) [:, 0].reshape(-1, 1)
            ys = np.array(self.dataset) [:, 1].reshape(-1, 1)

            # fit the GP with the updated dataset
            # Watch out: it's logarithmic because it's about the learning_rate
            if self.mode == "log":
                X_gp = np.log10(xs)
            else:
                X_gp = xs

            self.gp.fit(X_gp, ys)

            # calculate maximum utilization and its position
            x_max, util_max, util = self._max_acq()

            self.states.append({
                "dataset": self.dataset.copy(),
                "util": util,
                "GP": self.gp.predict(self.gp_search_space, return_std=True)
            })

            # hier müsste best_return_x und best_return_param angepasst werden, weil es sonst nicht gespeichert wird.
            
        return x_max, best_return_x, best_return_param

    
    def plot_iteration(self, iteration_idx):
        """
        Plotting one interation
        
        - points
        - gp posterior (mean + ±2σ)
        - acquisition funktion (WEI)

        iteration_idx: int
            Index in self.states
        """

        search_space = self.search_space.reshape(-1)
        
        if iteration_idx < 0 or iteration_idx >= len(self.states):
            raise IndexError(f"iteration_idx {iteration_idx} out of range (0..{len(self.states)-1})")

        state = self.states[iteration_idx]

        data = state["dataset"]
        util = state["util"].reshape(-1)
        gp = state["GP"]

        # create figure with two plots (ax1: GP fitting, ax2: utility function)
        figure = plt.figure(iteration_idx)
        gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[3, 1], figure=figure)
        ax1 = figure.add_subplot(gs[0])
        ax1.set_xticklabels([])  # turn off x labeling of upper plot
        ax1.set_title("Iteration %d" % iteration_idx)
        ax2 = figure.add_subplot(gs[1])

        # check if we need to set a logarithmic scale
        if self.mode == "log":
            ax1.set_xscale("log")
            ax2.set_xscale("log")

        mu, std = gp
        mu = mu.reshape(-1)
        ax1.plot(search_space, mu,
                 color="blue", label="GP mean")
        ax1.fill_between(search_space,
                         mu - (std * 1), mu + (std * 1),
                         color="blue", alpha=0.3, label="GP std")

        # plot the dataset
        xs = np.array(data)[:, 0]
        ys = np.array(data)[:, 1]

        # just choose one point of the duplicated (through sobol) 
        uniq_xs, idx = np.unique(xs, return_index=True)
        uniq_ys = ys[idx]
        
        ax1.scatter(uniq_xs, uniq_ys, color="blue", label="Dataset")

        # plot the utility function
        ax2.plot(search_space, util, color="green", label="Utility function")
        ax2.fill_between(search_space,
                         np.zeros_like(util),
                         util.reshape(-1), alpha=0.3, color="green")

        figure.legend(loc="lower center", ncol=4)

        plt.savefig(self.path+f"/")

        plt.show()

    def plot_all(self):
        """
        Plots all states/iterations made during optimization until now.
        """
        
        for i, state in enumerate(self.states):
            self.plot_iteration(i)


    def plot_all_in_one(self, cols, folder_path, seed):
        n = len(self.states)
        rows = (n + cols - 1) // cols
    
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
        axes = axes.flatten()
    
        for i, state in enumerate(self.states):
            ax = axes[i]
            search_space = self.search_space.reshape(-1)
    
            mu, std = state["GP"]
            util = state["util"].reshape(-1)
            xs = np.array(state["dataset"])[:, 0]
            ys = np.array(state["dataset"])[:, 1]
    
            # GP Mean
            ax.plot(search_space, mu, label="GP Mean")
            ax.fill_between(search_space, mu-std, mu+std, alpha=0.3)
    
            # Data
            ax.scatter(xs, ys, s=20, color="red")
    
            # Title
            ax.set_title(f"Iteration {i}")
            ax.set_xscale("log")
    
        # deactivate inactive axes
        for j in range(i+1, len(axes)):
            axes[j].axis("off")
    
        plt.tight_layout()
        plt.savefig(folder_path+f"Seed_{seed}_all_iterations.png", dpi=200)
        plt.close()

    