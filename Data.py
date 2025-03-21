import pandas as pd

def load_data(filename):
    data = pd.read_csv(filename)
    data['Date'] = pd.to_datetime(data['Date'])  # Konwersja na format daty
    return data['Date'].values, data['Close'].values