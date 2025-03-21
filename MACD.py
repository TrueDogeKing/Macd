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
    capital = initial_capital
    holdings = 0
    transactions = []
    
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

def plot_macd(dates, macd, signal, buy_signals, sell_signals, buy_values, sell_values):
    """
    Visualizes MACD, Signal line, and exact buy/sell points.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, macd, label='MACD', color='blue')
    plt.plot(dates, signal, label='Signal', color='red')

    # Use exact X (date) & Y (MACD value) for buy/sell markers
    # Sygnały kupna (zielone)
    plt.scatter(buy_signals, buy_values, marker='^', color='green', label='Buy Signal', zorder=3)

    # Sygnały sprzedaży (czerwone)
    plt.scatter(sell_signals, sell_values, marker='v', color='red', label='Sell Signal', zorder=3)

    plt.legend()
    plt.title('MACD & Signal Line')
    plt.xlabel('Date')
    plt.ylabel('MACD Value')
    plt.xticks(rotation=45)
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

print("Transaction Log:")
for t in transactions:
    print(f"{t[0]} - {t[1]} at {t[2]:.2f}, Holdings: {t[3]:.2f}")
print(f"Final Portfolio Value: {final_value:.2f}")

# Wizualizacja wyników
plot_macd(dates, macd, signal, buy_signals, sell_signals,buy_values, sell_values)
