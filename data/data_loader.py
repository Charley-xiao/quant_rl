import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataLoader:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))  # Normalizing data between 0 and 1

    def load_data_from_csv(self, file_path, date_col='Date', index_col=None):
        """
        Loads historical market data from a CSV file.
        :param file_path: Path to the CSV file.
        :param date_col: Column containing the date (default 'Date').
        :param index_col: Optional column to use as the index.
        :return: Pandas DataFrame of stock data.
        """
        data = pd.read_csv(file_path, parse_dates=[date_col], index_col=index_col)
        return data

    def load_data_from_yfinance(self, tickers, start_date, end_date=None, interval='1d'):
        """
        Loads real-time or historical market data from Yahoo Finance using yfinance.
        :param tickers: List of stock tickers to download.
        :param start_date: Start date for the data (format: 'YYYY-MM-DD').
        :param end_date: Optional end date for the data. If None, defaults to today.
        :param interval: Data interval ('1d', '1wk', '1mo', etc.).
        :return: Pandas DataFrame of stock data.
        """
        data = yf.download(tickers, start=start_date, end=end_date, interval=interval)
        return data

    def process_data(self, data, columns_to_normalize=None):
        """
        Preprocesses the data: normalize the selected columns and prepare for input to the model.
        :param data: Pandas DataFrame of stock prices or market data.
        :param columns_to_normalize: List of column names to normalize.
        :return: Normalized data as a NumPy array.
        """
        if columns_to_normalize is not None:
            data[columns_to_normalize] = self.scaler.fit_transform(data[columns_to_normalize])

        return data

    def split_data(self, data, train_ratio=0.7, val_ratio=0.15):
        """
        Splits the data into training, validation, and test sets.
        :param data: The input DataFrame to be split.
        :param train_ratio: The proportion of the data to use for training (default is 70%).
        :param val_ratio: The proportion of the data to use for validation (default is 15%).
        :return: Tuple of (train_data, val_data, test_data).
        """
        train_size = int(len(data) * train_ratio)
        val_size = int(len(data) * val_ratio)
        test_size = len(data) - train_size - val_size

        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]

        return train_data, val_data, test_data

    def inverse_transform(self, normalized_data):
        """
        Reverts normalized data back to the original scale.
        :param normalized_data: The normalized data to revert.
        :return: Data in its original scale.
        """
        return self.scaler.inverse_transform(normalized_data)
