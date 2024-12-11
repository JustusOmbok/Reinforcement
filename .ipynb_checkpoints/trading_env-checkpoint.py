# Gym stuff
import gym
import gym_anytrading
import MetaTrader5 as mt5

# Stable baselines - rl stuff
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C
from gym_anytrading.envs import ForexEnv
import numpy as np

# Processing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# Function to fetch historical data from MetaTrader5
def fetch_data(symbol, timeframe, start, end):
    rates = mt5.copy_rates_range(symbol, timeframe, start, end)
    if rates is None or len(rates) == 0:
        print(f"No data found for symbol {symbol} in the given date range.")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], unit='s')
    df['Date'] = pd.to_datetime(df['Date'])

    # Ensure volume is in the DataFrame
    if 'tick_volume' in df.columns:
        df['volume'] = df['tick_volume']
    elif 'real_volume' in df.columns:
        df['volume'] = df['real_volume']
    else:
        print("Volume not found")  # Default if volume is not available

    return df