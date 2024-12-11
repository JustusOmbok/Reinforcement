import MetaTrader5 as mt5
import numpy as np
import datetime
import pandas as pd
from stable_baselines3.common.callbacks import EvalCallback
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.monitor import Monitor
import indicators as ind
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.evaluation import evaluate_policy
from gym_anytrading.envs import ForexEnv
from gym_anytrading.envs.trading_env import Actions, Positions
from finta import TA
import optuna
from stable_baselines3.common.utils import set_random_seed
from sklearn.preprocessing import MinMaxScaler
import logging


# Function to fetch historical data from MetaTrader5
def fetch_data(symbol, timeframe, start, end, output_file="processed_data.csv"):
    rates = mt5.copy_rates_range(symbol, timeframe, start, end)
    if rates is None or len(rates) == 0:
        print(f"No data found for symbol {symbol} in the given date range.")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], unit='s')

    
    df['volume'] = df['tick_volume']

    df.rename(columns=lambda x: x.strip(), inplace=True)
    if df.isna().sum().any():
        df.ffill(inplace=True)  # Fill missing values forward

    # Rename columns to standard names
    df.rename(columns={'time': 'Date', 'open': 'Open', 'close': 'Close', 'volume': 'Volume', 'low': 'Low', 'high': 'High'}, inplace=True)
    df.drop(columns=['tick_volume', 'real_volume', 'spread'], inplace=True)
    
    # Save processed data to a CSV file
    df.to_csv(output_file)
    print(f"Processed data saved to {output_file}")
    
    return df

def create_features(df):

    df['sma_100'] = df['Close'].rolling(window=100).mean()
    df['sma_200'] = df['Close'].rolling(window=200).mean()
    df['ATR'] = ind.compute_atr(df['High'], df['Low'], df['Close'], window=14)
    df['rolling_std'] = df['Close'].rolling(window=20).std()
    df['slowk'], df['slowd'] = ind.compute_stochastic(df['High'], df['Low'], df['Close'])
    df['ROC'] = df['Close'].pct_change(periods=12)
    df['CCI'] = ind.compute_cci(df['High'], df['Low'], df['Close'], window=20)
    df['Williams_%R'] = ind.compute_williams_r(df['High'], df['Low'], df['Close'], window=14)
    df['MFI'] = ind.compute_mfi(df['High'], df['Low'], df['Close'], df['Volume'], window=14)
    df['SMA_12'] = TA.SMA(df, 12)  # Simple Moving Average
    df['EMA_50'] = TA.EMA(df, 50)  # Exponential Moving Average
    df['RSI'] = TA.RSI(df, 14)  # Relative Strength Index
    df['MACD'] = TA.MACD(df)['MACD']  # MACD line
    df['Signal_Line'] = TA.MACD(df)['SIGNAL']  # MACD signal line
    df['OBV'] = TA.OBV(df)  # On-Balance Volume
    bb = TA.BBANDS(df, period=20, std_multiplier=2)  # Bollinger Bands
    df['Upper_BB'] = pd.to_numeric(bb['BB_UPPER'], errors='coerce')
    df['Middle_BB'] = pd.to_numeric(bb['BB_MIDDLE'], errors='coerce')
    df['Lower_BB'] = pd.to_numeric(bb['BB_LOWER'], errors='coerce')
    df['keltner_upper'], df['keltner_lower'] = ind.compute_keltner_channels(df['Close'], df['High'], df['Low'])
    df['price_rsi_divergence'] = ind.detect_divergence(df['Close'], df['RSI'], lookback=5)
    df['ha_open'], df['ha_high'], df['ha_low'], df['ha_close'] = ind.compute_heikin_ashi(df)
    df['fisher_transform'] = ind.compute_fisher_transform(df['Close'])
    df['vwap'] = ind.compute_vwap(df['Close'], df['Volume'])
    df['eri_bull_power'], df['eri_bear_power'] = ind.compute_eri(df['High'], df['Low'], df['Close'])
    df['cmo'] = ind.compute_cmo(df['Close'])
    df['kvo'] = ind.compute_kvo(df['Close'], df['High'], df['Low'], df['Volume'])
    df['dpo'] = ind.compute_dpo(df['Close'])
    df['mfi_divergence'] = ind.detect_mfi_divergence(df['Close'], df['MFI'])

    return df

def normalize_indicators(df, columns_to_normalize):
    scaler = MinMaxScaler()
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df

def add_signals(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Low'].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['Low', 'Volume', 'SMA_12', 'EMA_50', 'RSI', 'MACD', 'Signal_Line', 'OBV', 'Upper_BB', 'Middle_BB', 'Lower_BB', 'sma_100', 'sma_200', 'ATR', 'rolling_std', 'slowk', 'slowd', 'ROC', 'CCI', 'Williams_%R', 'MFI', 'keltner_upper', 'keltner_lower', 'price_rsi_divergence', 'ha_open', 'ha_high', 'ha_low', 'ha_close', 'fisher_transform', 'vwap', 'eri_bull_power', 'eri_bear_power', 'cmo', 'kvo', 'dpo', 'mfi_divergence']].to_numpy()[start:end]
    return prices, signal_features

# Custom environment
class CustomForexEnv(ForexEnv):
    # Override _process_data to use add_signals
    _process_data = add_signals
    

# Main trading function
def main_trading_loop():

    symbol = "XAUUSD"
    timeframe = mt5.TIMEFRAME_D1
    start = datetime.datetime(2024, 1, 1)
    end = datetime.datetime.now()

    
    df = fetch_data(symbol, timeframe, start, end)
    df = pd.read_csv('processed_data.csv')
    df.drop(columns=['Unnamed: 0'], inplace=True)
    df.set_index('Date', inplace=True)
    df['Volume'] = df['Volume'].apply(lambda x: float(str(x).replace(",", "")))
    df = create_features(df)
    #df = normalize_indicators(df, df.columns)
    df.fillna(0, inplace=True)
    
    
    env2 = CustomForexEnv(df=df, window_size=12, frame_bound=(12,len(df)))
    #print(env2.df.head())
    env_maker = lambda: env2
    env = VecNormalize(DummyVecEnv([env_maker]), norm_obs=True, norm_reward=True)
    model = RecurrentPPO(
        "MlpLstmPolicy", 
        env,
        learning_rate=0.0001,
        verbose=1
    )
    model.learn(total_timesteps=100000)
    model.save("ppo_forex_model")

    env_eval = CustomForexEnv(df=df, window_size=12, frame_bound=(80,250))
    obs, _ = env_eval.reset()  # Assuming reset returns a tuple (observation, additional info)
    while True: 
        obs = obs[np.newaxis, ...]  # Add a new axis to the observation to make it batch-like
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            print("info", info)
            break


if __name__ == "__main__":
    if not mt5.initialize(login=100003703, server="FBS-Demo", password="jV9}8d,)"):
        print("Failed to initialize MetaTrader5")
        mt5.shutdown()
        quit()
    
    main_trading_loop()
    mt5.shutdown()