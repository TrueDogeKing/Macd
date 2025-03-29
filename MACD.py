import numpy as np
import matplotlib.pyplot as plt
from Data import load_data

def calculate_ema(prices, period):
    """
    Oblicza wykładniczą średnią kroczącą (EMA) bez użycia zewnętrznych bibliotek (poza numpy)
    """
    alpha = 2 / (period + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]  # Inicjalizacja pierwszej wartości EMA jako pierwszej ceny
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    return ema

def calculate_macd(prices):
    """
    Oblicza wskaźnik MACD i linię sygnału (SIGNAL) zgodnie z wytycznymi projektu
    """
    ema12 = calculate_ema(prices, 12)
    ema26 = calculate_ema(prices, 26)
    macd = ema12 - ema26
    signal = calculate_ema(macd, 9)
    return macd, signal

def find_trade_signals(dates, prices,macd, signal):
    """
    Finds exact buy/sell crossover points with precise date & MACD value.
    """
    buy_signals = []
    sell_signals = []
    buy_values = []
    sell_values = []

    buy_prices, sell_prices = [], []
    
    # Convert dates to numeric values for interpolation
    dates_numeric = (dates - dates[0]) / np.timedelta64(1, 's')  # Convert to seconds

    for i in range(1, len(macd)):
        if (macd[i - 1] < signal[i - 1] and macd[i] > signal[i]) or (macd[i - 1] > signal[i - 1] and macd[i] < signal[i]):
            # Find exact crossing fraction 't'
            t = (signal[i - 1] - macd[i - 1]) / ((macd[i] - macd[i - 1]) - (signal[i] - signal[i - 1]))

            # Interpolate exact timestamp
            crossing_time_numeric = dates_numeric[i - 1] + t * (dates_numeric[i] - dates_numeric[i - 1])
            crossing_time = dates[0] + np.timedelta64(int(crossing_time_numeric), 's')

            # Interpolate exact MACD value at crossing
            crossing_value = macd[i - 1] + t * (macd[i] - macd[i - 1])
            crossing_price= prices[i - 1] + t * (prices[i] - prices[i - 1])

            if macd[i - 1] < signal[i - 1] and macd[i] > signal[i]:  
                buy_signals.append(crossing_time)
                buy_values.append(crossing_value)
                buy_prices.append(crossing_price)
            elif macd[i - 1] > signal[i - 1] and macd[i] < signal[i]:  
                sell_signals.append(crossing_time)
                sell_values.append(crossing_value)
                sell_prices.append(crossing_price)

    return np.array(buy_signals), np.array(sell_signals), np.array(buy_values), np.array(sell_values),np.array(buy_prices), np.array(sell_prices)


def simulate_trading(buy_signals, sell_signals, buy_prices, sell_prices, initial_capital=1000):
    capital = 0
    holdings = initial_capital
    transactions = []

    transactions = [(dates[0], "BUY", 6.02, 1000)] # dodanie initial capital do log
    for i in range(len(buy_signals)):
        if capital > 0:
            holdings = capital / buy_prices[i]
            capital = 0
            transactions.append((buy_signals[i], 'BUY', buy_prices[i], holdings))
        if i < len(sell_signals):
            capital = holdings * sell_prices[i]
            holdings = 0
            transactions.append((sell_signals[i], 'SELL', sell_prices[i], capital))
    
    final_value = capital if capital > 0 else holdings * sell_prices[-1]
    return transactions, final_value

def plot_macd_with_price(dates, prices, macd, signal, buy_signals, sell_signals, buy_values, sell_values, buy_prices, sell_prices):
    """
    Visualizes MACD, Signal line, and Stock Price in subplots with buy/sell points.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # MACD and Signal Line
    ax1.plot(dates, macd, label='MACD', color='blue')
    ax1.plot(dates, signal, label='Signal', color='red')
    ax1.scatter(buy_signals, buy_values, marker='^', color='green', label='Buy Signal', zorder=3)
    ax1.scatter(sell_signals, sell_values, marker='v', color='red', label='Sell Signal', zorder=3)
    ax1.set_ylabel("MACD Value")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_title('MACD & Signal Line')

    # Stock Price
    ax2.plot(dates, prices, label='Stock Price', color='black', alpha=0.7)
    ax2.bar(buy_signals, buy_prices, color='green', alpha=0.3, width=0.5, label='Buy Price')
    ax2.bar(sell_signals, sell_prices, color='red', alpha=0.3, width=0.5, label='Sell Price')
    ax2.set_ylabel("Stock Price")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.set_title('Stock Price & Buy/Sell Signals')
    
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.show()



def plot_portfolio_value(transactions):
    """
    Plots portfolio value (capital + stock holdings) over time after each transaction.
    """
    times = []
    portfolio_values = []
    capital = 0
    holdings = 0

    for t in transactions:
        times.append(t[0])  # Transaction timestamp
        if t[1] == "BUY":
            holdings = t[3]  # Update stock holdings
            capital = 0  # Spent cash
        elif t[1] == "SELL":
            capital = t[3]  # Update cash
            holdings = 0  # Sold stocks
        # Calculate portfolio value (cash + stock holdings)
        portfolio_values.append(capital + holdings * (t[2] if t[1] == "SELL" else t[2]))

    plt.figure(figsize=(12, 5))
    plt.plot(times, portfolio_values, marker='o', linestyle='-', color='blue', label="Portfolio Value")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.title("Portfolio Value Over Time")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

def plot_holdings_over_time(transactions):
    """
    Plots the number of stocks owned over time, skipping SELL transactions (only when holdings > 0).
    """
    times = []
    holdings_over_time = []

    for t in transactions:
        if t[1] == "BUY":  # Only consider buy transactions
            times.append(t[0])  # Transaction timestamp
            holdings_over_time.append(t[3])  # Number of shares bought

    plt.figure(figsize=(12, 5))
    plt.plot(times, holdings_over_time, marker='o', linestyle='-', color='green', label="Stock Holdings")
    plt.xlabel("Date")
    plt.ylabel("Number of Shares Owned")
    plt.title("Stock Holdings Over Time (BUY Transactions Only)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


# Wczytanie danych z pliku CSV za pomocą modułu Data.py
dates, prices = load_data('nvidia.csv')
dates = dates[::-1]  
prices = prices[::-1]  
# prices = np.random.rand(100) * 100  # Symulowane ceny akcji
# Obliczenia
macd, signal = calculate_macd(prices)
buy_signals, sell_signals,buy_values, sell_values,buy_prices,sell_prices= find_trade_signals(dates,prices,macd, signal)

transactions, final_value = simulate_trading(buy_signals, sell_signals, buy_prices, sell_prices)

profitable_trades=0
print("Transaction Log:")
for i in range(0, len(transactions),2):  
    buy = transactions[i]
    sell = transactions[i + 1] if i + 1 < len(transactions) else None
    
    if sell:
        profit = sell[2] - buy[2]  # Obliczenie zysku/straty
        print(f"{buy[0]} - {buy[1]} at {buy[2]:.2f}, Holdings: {buy[3]:.2f}")
        print(f"{sell[0]} - {sell[1]} at {sell[2]:.2f}, Capital: {sell[3]:.2f}, Profit: {profit:.2f}")
        if profit > 0:
            profitable_trades += 1

print(f"\nFinal Portfolio Value: {final_value:.2f}")
print(f"Number of Profitable Transactions: {profitable_trades}")


# Wizualizacja wyników
plot_macd_with_price(dates, prices, macd, signal, buy_signals, sell_signals, buy_values, sell_values, buy_prices, sell_prices)
plot_portfolio_value(transactions)
plot_holdings_over_time(transactions)



