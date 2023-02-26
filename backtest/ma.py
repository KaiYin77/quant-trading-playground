import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def fetch_historical_price_data(symbol, timeframe, days=365, limit=1000):
    exchange = ccxt.binance()
    start_date = datetime.now() - timedelta(days)
    end_date = datetime.now()
    ohlcv = exchange.fetch_ohlcv(
            symbol, 
            timeframe, 
            exchange.parse8601(str(start_date)), exchange.parse8601(str(end_date)))

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def calculate_metrics(df, ma_window, initial_balance):
    # Calculate the moving average
    ma = df['close'].rolling(window=ma_window).mean()

    # Define the strategy
    trades = []
    in_position = False
    for i in range(len(df)):
        if df['close'][i] > ma[i] and not in_position:
            # Buy signal
            trades.append({'timestamp': df.index[i], 'action': 'buy', 'price': df['close'][i]})
            in_position = True
        elif df['close'][i] < ma[i] and in_position:
            # Sell signal
            trades.append({'timestamp': df.index[i], 'action': 'sell', 'price': df['close'][i]})
            in_position = False

    # Simulate trades
    balance = initial_balance
    eth_quantity = 0
    trade_history = []
    for trade in trades:
        if trade['action'] == 'buy':
            eth_quantity = balance / trade['price']
            balance = 0
        elif trade['action'] == 'sell':
            balance = eth_quantity * trade['price']
            eth_quantity = 0
        trade_history.append({'timestamp': trade['timestamp'], 'action': trade['action'], 'price': trade['price'], 'balance': balance, 'eth_quantity': eth_quantity})

    # Calculate total return
    total_return = (balance + eth_quantity * df.iloc[-1]['close']) / initial_balance - 1

    # Calculate volatility
    daily_returns = df['close'].pct_change()
    volatility = daily_returns.std() * np.sqrt(len(df))

    # Calculate Sharpe ratio
    risk_free_rate = 0.0
    daily_risk_premium = daily_returns - risk_free_rate
    sharpe_ratio = (daily_risk_premium.mean() / daily_risk_premium.std()) * np.sqrt(252)

    # Calculate maximum drawdown
    cum_returns = (daily_returns + 1).cumprod()
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    return {
        'total_return': total_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'trade_history': trade_history,
        'ma': ma
    }

def plot_action(ma_window, df, metrics):
    # Plot the chart
    plt.figure(figsize=(16, 8))
    plt.plot(df['close'], label='ETH/USDT')
    plt.plot(metrics['ma'], label=f'{ma_window}-day MA')
    
    buy_points = [trade['timestamp'] for trade in metrics['trade_history'] if trade['action'] == 'buy']
    sell_points = [trade['timestamp'] for trade in metrics['trade_history'] if trade['action'] == 'sell']
    
    plt.plot(buy_points, df.loc[buy_points]['close'], '^', markersize=10, color='green', label='Buy')
    plt.plot(sell_points, df.loc[sell_points]['close'], 'v', markersize=10, color='red', label='Sell')
    
    plt.legend()
    plt.show()


# Example usage
symbol = 'ETH/USDT'
timeframe = '4h'
limit = 1000
initial_balance = 500
ma_window = 7
days = 90

df = fetch_historical_price_data(symbol, timeframe, days, limit)
metrics = calculate_metrics(df, ma_window, initial_balance)

print(f'Total return: {metrics["total_return"]:.2%}')
print(f'Volatility: {metrics["volatility"]:.2%}')
print(f'Sharpe ratio: {metrics["sharpe_ratio"]:.2f}')
print(f'Max drawdown: {metrics["max_drawdown"]:.2f}')

plot_action(ma_window, df, metrics)

