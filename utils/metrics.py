import numpy as np

class Metrics:
    @staticmethod
    def annual_return(portfolio_values, trading_days_per_year=252):
        """
        Computes the Annual Return (CAGR).
        :param portfolio_values: A list or array of portfolio values over time.
        :param trading_days_per_year: Number of trading days in a year (default: 252).
        :return: The annual return as a float.
        """
        total_return = portfolio_values[-1] / portfolio_values[0] - 1
        num_years = len(portfolio_values) / trading_days_per_year
        return (1 + total_return) ** (1 / num_years) - 1

    @staticmethod
    def maximum_drawdown(portfolio_values):
        """
        Computes the Maximum Drawdown (MDD).
        :param portfolio_values: A list or array of portfolio values over time.
        :return: The maximum drawdown as a float.
        """
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        return np.min(drawdown)

    @staticmethod
    def sharpe_ratio(portfolio_returns, risk_free_rate=0, trading_days_per_year=252):
        """
        Computes the Sharpe Ratio.
        :param portfolio_returns: A list or array of daily portfolio returns.
        :param risk_free_rate: The risk-free rate (default: 0).
        :param trading_days_per_year: Number of trading days in a year (default: 252).
        :return: The Sharpe Ratio as a float.
        """
        excess_returns = portfolio_returns - risk_free_rate / trading_days_per_year
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(trading_days_per_year)

    @staticmethod
    def short_term_risk(portfolio_returns):
        """
        Computes Short-term Portfolio Risk (standard deviation of returns).
        :param portfolio_returns: A list or array of portfolio returns over time.
        :return: The short-term portfolio risk as a float.
        """
        return np.std(portfolio_returns)
