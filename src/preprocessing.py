import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, kpss



def calculate_data_index(data, window, data_price_type, risk_free_rate):
    
    #Calculates return and risk metrics for each symbol in the data dictionary.

    results = {}

    for symbol, df in data.items():
        df = df.copy()  # Work on a copy to avoid modifying original

        # Fill missing data
        df[data_price_type] = df[data_price_type].fillna(method="ffill")

        # Simple return
        df["simple_return"] = df[data_price_type].pct_change()

        # Log return
        df["log_return"] = np.log(df[data_price_type] / df[data_price_type].shift(1))

        # Cumulative return
        df["cumulative_return"] = (1 + df["simple_return"]).cumprod() - 1

        # Rolling average return
        df["rolling_avg_return"] = df["log_return"].rolling(window).mean()

        # Rolling volatility
        df["rolling_volatility"] = df["log_return"].rolling(window).std()

        # Rolling Sharpe Ratio
        df["rolling_sharpe"] = (df["rolling_avg_return"] - risk_free_rate) / df["rolling_volatility"]

        # Max Drawdown
        cumulative = (1 + df["simple_return"]).cumprod()
        df["rolling_max"] = cumulative.cummax()
        df["drawdown"] = (cumulative - df["rolling_max"]) / df["rolling_max"]
        df["max_drawdown"] = df["drawdown"].rolling(window).min()

        # Drop missing rows caused by rolling calculations
        df.dropna(inplace=True)

        # Save results
        results[symbol] = df

    return results



def plot_data_index(data_index, symbol, column, color):
    # Check if symbol exists
    if symbol not in data_index:
        print(f" Symbol '{symbol}' not found in metrics.")
        print(f" Available symbols: {list(data_index.keys())}")
        return

    df = data_index[symbol]

    # Check if column exists in that symbol's DataFrame
    if column not in df.columns:
        print(f" Column '{column}' not found in DataFrame for {symbol}.")
        print(f" Available columns: {df.columns.tolist()}")
        return

    # Plot the data
    df[column].plot(label=symbol, color=color, figsize=(16, 6))
    plt.xlabel('Date')
    plt.ylabel(column.capitalize())
    plt.title(f'{symbol} ({column})')
    plt.grid()
    plt.legend()
    plt.show()



def adf_test(series):
    try:
        if series.nunique() <= 1:
            return False
        result = adfuller(series.dropna())
        return result[1] < 0.05
    except:
        return False

def kpss_test(series):
    try:
        if series.nunique() <= 1:
            return False
        result = kpss(series.dropna(), regression='c', nlags='auto')
        return result[1] >= 0.05
    except:
        return False

# Data preprocessing function for ML
def data_preprocessing(data, num_lags, train_test_split):
    x, y = [], []
    for i in range(len(data) - num_lags):
        x.append(data[i:i + num_lags])
        y.append(data[i + num_lags])
    x = np.array(x)
    y = np.array(y)
    split_index = int(train_test_split * len(x))
    return x[:split_index], y[:split_index], x[split_index:], y[split_index:]
