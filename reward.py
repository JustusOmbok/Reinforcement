from main import CustomForexEnv  # Import the custom Forex environment
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv
import datetime
from main import get_preprocessed_data  # Import reusable functions

# Initialize MetaTrader5
if not mt5.initialize(login=100003703, server="FBS-Demo", password="jV9}8d,)"):
    print("Failed to initialize MetaTrader5")
    mt5.shutdown()
    quit()

# Fetch and preprocess data
symbol = "XAUUSD"
timeframe = mt5.TIMEFRAME_H1
start = datetime.datetime(2024, 1, 1)
end = datetime.datetime.now()

try:
    df = get_preprocessed_data(symbol, timeframe, start, end)
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected a DataFrame, but got {type(df).__name__}")
    print(f"Data fetched for symbol {symbol} with shape: {df.shape}")
    if df.empty:
        raise ValueError(f"No data found for symbol {symbol} in the given date range.")
except Exception as e:
    print(f"Error fetching data: {e}")
    mt5.shutdown()
    quit()

# Load or create a model
env = DummyVecEnv([lambda: CustomForexEnv(df, window_size=12, frame_bound=(12,50))])
model = RecurrentPPO.load("ppo_forex_model")

# Debugging information
#print(f"Frame bound: {env.get_attr('frame_bound')[0]}")
##print(f"Prices length: {len(env.get_attr('prices')[0])}")

# Run the agent
obs = env.reset()
while True: 
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        print("info", info)
        break

# Shutdown MetaTrader5
mt5.shutdown()
