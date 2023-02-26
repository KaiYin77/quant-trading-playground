import time
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

import talib
import ccxt

import matplotlib.pyplot as plt

def get_historical_data(symbol, timeframe, days):
    exchange = ccxt.binance()
    start_date = datetime.now() - timedelta(days)
    end_date = datetime.now()
    # Fetch the historical data from the exchange
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, exchange.parse8601(str(start_date)), exchange.parse8601(str(end_date)))

    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # Convert the timestamp to a datetime object
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Keep only the columns we need
    df.set_index('timestamp', inplace=True)

    return df


def calculate_fibonacci_levels(df):
    # Calculate the Fibonacci retracement levels
    high = df['high'].max()
    low = df['low'].min()
    diff = high - low
    levels = [high - (0.236 * diff), high - (0.382 * diff), high - (0.5 * diff), high - (0.618 * diff), high - (0.786 * diff)]
    
    return levels


def create_signals(df, levels):
    # Create signals based on the retracement levels
    df['signal'] = np.where(df['close'] <= levels[0], 1, 0)  # Buy signal
    df['signal'] = np.where(df['close'] >= levels[-1], -1, df['signal'])  # Sell signal

    return df


def backtest_strategy(df):
    # Backtest the strategy
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['signal'].shift(1) * df['returns']
    df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()

    return df


def calculate_metrics(df):
    # Calculate metrics
    volatility = df['returns'].std() * np.sqrt(252)
    sharp_ratio = (df['strategy_returns'].mean() / df['strategy_returns'].std()) * np.sqrt(252)
    max_drawdown = (df['cumulative_returns'].cummax() - df['cumulative_returns']) / df['cumulative_returns'].cummax()
    max_drawdown = max_drawdown.max()
    
    final_portfolio_value = df['portfolio_value'][-1]
    total_returns = (final_portfolio_value - initial_capital) / initial_capital

    return total_returns, volatility, sharp_ratio, max_drawdown


def calculate_portfolio_value(df, initial_capital):
    # Set up position size
    position_size = initial_capital / df['close'][0]

    # Calculate the portfolio value
    df['portfolio_value'] = position_size * df['cumulative_returns'] * df['close']

    # Calculate the cash balance
    df['cash_balance'] = initial_capital - (df['portfolio_value'] - (df['close'] * position_size))

    return df

def plot_fibonacci_results(df, levels):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)

    # Plot ETH prices and Fibonacci levels
    ax1.plot(df['close'], label='ETH Price')
    for i, level in enumerate(levels):
        ax1.axhline(level, linestyle='--', alpha=0.7, color='grey', label=f'Fib Level {i+1}')
    ax1.legend()

    # Plot portfolio value and buy/sell signals
    ax2.plot(df['portfolio_value'], label='Portfolio Value')
    ax2.plot(df['cash_balance'], label='Cash Balance')
    ax2.plot(df[df['signal'] == 1].index, df['portfolio_value'][df['signal'] == 1], marker='^', color='g', markersize=10, label='Buy Signal')
    ax2.plot(df[df['signal'] == -1].index, df['portfolio_value'][df['signal'] == -1], marker='v', color='r', markersize=10, label='Sell Signal')
    ax2.legend()

    plt.show()

def save_to_csv(df):
    from pathlib import Path  
    file_path = Path('backtest/records/') / Path(time.strftime("%Y%m%d%H%M") + '.csv')
    file_path.parent.mkdir(parents=True, exist_ok=True)  
    df.to_csv(file_path)

if __name__ == '__main__':
    symbol = 'ETH/USDT'
    timeframe = '4h'
    initial_capital = 500
    days = 90

    df = get_historical_data(symbol, timeframe, days)
    levels = calculate_fibonacci_levels(df)
    df = create_signals(df, levels)
    df = backtest_strategy(df)
    df = calculate_portfolio_value(df, initial_capital)
    total_return, volatility, sharpe_ratio, max_drawdown = calculate_metrics(df)

    print(f'Total return: {total_return:.2%}')
    print(f'Volatility: {volatility:.2%}')
    print(f'Sharpe ratio: {sharpe_ratio:.2f}')
    print(f'Max drawdown: {max_drawdown:.2f}')
    
    plot_fibonacci_results(df, levels)
    save_to_csv(df)

