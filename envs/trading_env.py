import numpy as np

class TradingEnv:
    def __init__(self, data, initial_balance=1000000, transaction_cost=0.001):
        """
        Initializes the trading environment.
        :param data: A DataFrame or NumPy array of asset prices (rows: time, columns: assets).
        :param initial_balance: The initial portfolio balance.
        :param transaction_cost: Cost of transactions as a fraction (e.g., 0.001 = 0.1% per trade).
        """
        self.data = data
        self.n_assets = data.shape[1]
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost

        # Initialize variables
        self.current_step = 0
        self.balance = initial_balance
        self.portfolio_value = initial_balance
        self.portfolio_weights = np.array([1.0 / self.n_assets] * self.n_assets)  # Equal initial allocation
        self.portfolio = np.zeros(self.n_assets)
        self.done = False

    def reset(self):
        """
        Resets the environment to the initial state.
        :return: Initial state (prices of all assets at step 0).
        """
        self.current_step = 0
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.portfolio_weights = np.array([1.0 / self.n_assets] * self.n_assets)  # Equal initial allocation
        self.portfolio = np.zeros(self.n_assets)
        self.done = False
        return self._get_state()

    def step(self, action):
        """
        Executes the action (portfolio rebalancing) and moves the environment to the next step.
        :param action: New portfolio weights (should sum to 1).
        :return: Next state, reward, done, and additional information.
        """
        # Check if the episode is done
        if self.current_step >= len(self.data) - 1:
            self.done = True

        # Calculate the reward based on the previous portfolio
        reward = self.calculate_reward(action)

        # Update portfolio weights based on the action
        self.portfolio_weights = action

        # Move to the next step (next day's prices)
        self.current_step += 1

        # Get the next state
        next_state = self._get_state()

        return next_state, reward, self.done, {}

    def calculate_reward(self, new_weights):
        """
        Calculates the reward based on the portfolio's performance.
        :param new_weights: New portfolio weights after rebalancing.
        :return: Reward (e.g., portfolio return minus transaction costs).
        """
        current_prices = self.data[self.current_step]
        next_prices = self.data[self.current_step + 1]

        # Portfolio returns (price change from the current step to the next step)
        portfolio_return = np.dot(new_weights, (next_prices / current_prices - 1))

        # Transaction cost (based on the portfolio rebalancing)
        transaction_cost = self.transaction_cost * np.sum(np.abs(new_weights - self.portfolio_weights))

        # Reward is the net return after transaction costs
        reward = portfolio_return - transaction_cost

        return reward

    def _get_state(self):
        """
        Gets the current state (current asset prices).
        :return: Current asset prices.
        """
        return self.data[self.current_step]
