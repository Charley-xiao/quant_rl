import numpy as np

class Portfolio:
    def __init__(self, initial_balance=1000000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.portfolio_value = initial_balance
        self.asset_holdings = None

    def rebalance(self, new_weights, asset_prices):
        """
        Rebalance the portfolio according to the new weights.
        :param new_weights: Array of new portfolio weights (should sum to 1).
        :param asset_prices: Current prices of the assets.
        :return: Updated portfolio value.
        """
        total_value = self.balance
        #print(f'{total_value=}')
        self.asset_holdings = total_value * new_weights / asset_prices
        #print(f'{self.asset_holdings=}')

        # Update portfolio value
        self.portfolio_value = np.sum(self.asset_holdings * asset_prices)
        #print(f'{self.portfolio_value=}')

        return self.portfolio_value

    def get_portfolio_value(self, asset_prices):
        """
        Calculates the current portfolio value based on asset holdings and prices.
        :param asset_prices: Current prices of the assets.
        :return: Portfolio value.
        """
        if self.asset_holdings is None:
            return self.initial_balance
        self.portfolio_value = np.sum(self.asset_holdings * asset_prices)
        return self.portfolio_value
