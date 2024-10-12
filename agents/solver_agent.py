import numpy as np
from scipy.optimize import minimize

class SolverAgent:
    def __init__(self, n_assets, risk_function, bounds=None, options=None):
        self.n_assets = n_assets
        self.risk_function = risk_function  # A function to calculate risk based on asset weights
        self.bounds = bounds if bounds is not None else [(0, 1)] * n_assets  # Weights must be between 0 and 1
        self.options = options if options is not None else {'maxiter': 100}

    def optimize_risk(self, initial_weights):
        """
        Uses Particle Swarm Optimization (PSO) or another optimization method 
        to find the optimal portfolio weights that minimize the risk.
        """

        # Constraints: sum of weights should be 1 (i.e., a fully invested portfolio)
        constraints = ({
            'type': 'eq',
            'fun': lambda weights: np.sum(weights) - 1
        })

        # Initial guess
        if initial_weights is None:
            initial_weights = np.array([1.0 / self.n_assets] * self.n_assets)

        # Use Scipy's minimize function with a Sequential Least Squares Programming (SLSQP) method
        result = minimize(self.risk_function, initial_weights, method='SLSQP',
                          bounds=self.bounds, constraints=constraints,
                          options=self.options)

        # Optimal weights
        optimized_weights = result.x

        return optimized_weights

    def adjust_portfolio(self, current_portfolio, optimized_weights):
        """
        Adjust the current portfolio allocations based on the optimized weights.
        """
        # Assuming the current_portfolio is a dictionary with asset names as keys and their current weights as values
        new_portfolio = {}
        asset_names = list(current_portfolio.keys())

        for i, asset in enumerate(asset_names):
            new_portfolio[asset] = optimized_weights[i]

        return new_portfolio
