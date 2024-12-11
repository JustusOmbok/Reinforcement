import MetaTrader5 as mt5
import numpy as np
import datetime
import gymnasium as gym
import pandas as pd
from stable_baselines3.common.callbacks import EvalCallback
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.evaluation import evaluate_policy
from gym_anytrading.envs import ForexEnv
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

    df.set_index('time', inplace=True)
    df.rename(columns={'open': 'Open', 'close': 'Close', 'volume': 'Volume', 'low': 'Low', 'high': 'High'}, inplace=True)
    df.drop(columns=['tick_volume', 'real_volume', 'spread'], inplace=True)
    
    # Save processed data to a CSV file
    df.to_csv(output_file)
    print(f"Processed data saved to {output_file}")
    
    return df

# Main feature creation function with all updates
def create_features(df):
    # Existing indicators
    df['SMA'] = TA.SMA(df, 12)
    df['RSI'] = TA.RSI(df)
    df['OBV'] = TA.OBV(df)
    """df['sma_100'] = df['close'].rolling(window=100).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    df['RSI'] = compute_rsi(df['close'], window=14)
    df['ATR'] = compute_atr(df['high'], df['low'], df['close'], window=14)
    df['rolling_std'] = df['close'].rolling(window=20).std()
    df['slowk'], df['slowd'] = compute_stochastic(df['high'], df['low'], df['close'])
    df['ROC'] = df['close'].pct_change(periods=12)
    df['CCI'] = compute_cci(df['high'], df['low'], df['close'], window=20)
    df['Williams_%R'] = compute_williams_r(df['high'], df['low'], df['close'], window=14)
    df['MFI'] = compute_mfi(df['high'], df['low'], df['close'], df['volume'], window=14)
    
    # New indicators
    # Keltner Channels
    df['keltner_upper'], df['keltner_lower'] = compute_keltner_channels(df['close'], df['high'], df['low'])

    # Divergence Detection (between Price and RSI)
    df['price_rsi_divergence'] = detect_divergence(df['close'], df['RSI'], lookback=5)

    # Heikin-Ashi Candlesticks
    df['ha_open'], df['ha_high'], df['ha_low'], df['ha_close'] = compute_heikin_ashi(df)

    df['fisher_transform'] = compute_fisher_transform(df['close'])
    df['vwap'] = compute_vwap(df['close'], df['volume'])
    df['eri_bull_power'], df['eri_bear_power'] = compute_eri(df['high'], df['low'], df['close'])
    df['cmo'] = compute_cmo(df['close'])
    df['kvo'] = compute_kvo(df['close'], df['high'], df['low'], df['volume'])
    df['dpo'] = compute_dpo(df['close'])
    df['mfi_divergence'] = detect_mfi_divergence(df['close'], df['MFI'])"""

    df.dropna(inplace=True)
    return df

# Helper functions for indicators
def compute_dpo(close, period=20):
    sma = close.shift(int((period / 2) + 1)).rolling(window=period).mean()
    dpo = close - sma
    return dpo
def compute_kvo(close, high, low, volume, short_period=34, long_period=55):
    trend = ((high + low + close) / 3) - ((high.shift(1) + low.shift(1) + close.shift(1)) / 3)
    volume_force = trend * volume
    short_kvo = volume_force.ewm(span=short_period, adjust=False).mean()
    long_kvo = volume_force.ewm(span=long_period, adjust=False).mean()
    kvo = short_kvo - long_kvo
    return kvo
def detect_mfi_divergence(close, mfi, lookback=5):
    price_highs = close.rolling(window=lookback).max()
    price_lows = close.rolling(window=lookback).min()
    mfi_highs = mfi.rolling(window=lookback).max()
    mfi_lows = mfi.rolling(window=lookback).min()
    divergence = np.where((close > price_highs.shift(1)) & (mfi < mfi_highs.shift(1)), 1,
                          np.where((close < price_lows.shift(1)) & (mfi > mfi_lows.shift(1)), -1, 0))
    return pd.Series(divergence, index=close.index)

def compute_cmo(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).sum()
    loss = -delta.where(delta < 0, 0).rolling(window=period).sum()
    cmo = 100 * (gain - loss) / (gain + loss)
    return cmo

def compute_fisher_transform(close, period=10):
    min_low = close.rolling(window=period).min()
    max_high = close.rolling(window=period).max()
    epsilon = 1e-10  # Small value to avoid division by zero
    value = 2 * ((close - min_low) / (max_high - min_low + epsilon) - 0.5)
    value = np.clip(value, -0.999, 0.999)  # Clip values to avoid log of zero
    fisher_transform = (np.log((1 + value) / (1 - value + epsilon))).rolling(window=2).mean()
    return fisher_transform

def compute_vwap(close, volume):
    cum_volume_price = (close * volume).cumsum()
    cum_volume = volume.cumsum()
    vwap = cum_volume_price / cum_volume
    return vwap

def compute_eri(high, low, close, ema_window=13):
    ema = close.ewm(span=ema_window, adjust=False).mean()
    bull_power = high - ema
    bear_power = low - ema
    return bull_power, bear_power

def compute_keltner_channels(close, high, low, ema_window=20, atr_window=10, atr_multiplier=2):
    ema = close.ewm(span=ema_window, adjust=False).mean()
    atr = compute_atr(high, low, close, window=atr_window)
    upper_band = ema + (atr_multiplier * atr)
    lower_band = ema - (atr_multiplier * atr)
    return upper_band, lower_band

def detect_divergence(price, indicator, lookback=5):
    price_highs = price.rolling(window=lookback).max()
    price_lows = price.rolling(window=lookback).min()
    indicator_highs = indicator.rolling(window=lookback).max()
    indicator_lows = indicator.rolling(window=lookback).min()
    divergence = np.where((price > price_highs.shift(1)) & (indicator < indicator_highs.shift(1)), 1,
                          np.where((price < price_lows.shift(1)) & (indicator > indicator_lows.shift(1)), -1, 0))
    return pd.Series(divergence, index=price.index)

def compute_heikin_ashi(df):
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_open = (df['open'].shift(1) + df['close'].shift(1)) / 2
    ha_high = pd.concat([df['high'], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([df['low'], ha_open, ha_close], axis=1).min(axis=1)
    return ha_open, ha_high, ha_low, ha_close

def compute_rsi(series, window=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_atr(high, low, close, window=14):
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

def compute_stochastic(high, low, close, window=14):
    lowest_low = low.rolling(window=window).min()
    highest_high = high.rolling(window=window).max()
    slowk = 100 * (close - lowest_low) / (highest_high - lowest_low)
    slowd = slowk.rolling(window=3).mean()
    return slowk, slowd

def compute_cci(high, low, close, window=20):
    tp = (high + low + close) / 3
    sma = tp.rolling(window=window).mean()
    mad = tp.rolling(window=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma) / (0.015 * mad)
    return cci

def compute_williams_r(high, low, close, window=14):
    highest_high = high.rolling(window=window).max()
    lowest_low = low.rolling(window=window).min()
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return williams_r

def compute_mfi(high, low, close, volume, window=14):
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    positive_flow_sum = positive_flow.rolling(window=window).sum()
    negative_flow_sum = negative_flow.rolling(window=window).sum()
    money_flow_index = 100 - (100 / (1 + (positive_flow_sum / negative_flow_sum)))
    return money_flow_index

# Fetch and preprocess data
def get_preprocessed_data(symbol, timeframe, start, end):
    df = fetch_data(symbol, timeframe, start, end)
    if df.empty:
        raise ValueError("No data fetched.")
    df = create_features(df)  # Apply feature engineering
    
    return df

def add_signals(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Low'].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['Low', 'Volume','SMA', 'RSI', 'OBV']].to_numpy()[start:end]
    
    # Normalize the signal features
    scaler = MinMaxScaler()
    signal_features = scaler.fit_transform(signal_features)
    
    return prices, signal_features

# Custom environment
class CustomForexEnv(ForexEnv):
    _process_data = add_signals

# Create and train the model
def train_model(df):
    """best_params = {
        'n_steps': 316,
        'gamma': 0.9374683601508613,
        'learning_rate': 1.0617248050022004e-05,
        'ent_coef': 0.0023144315215881334,
        'clip_range': 0.2938081246524669,
        'n_epochs': 7,
        'gae_lambda': 0.8328470822125699,
        'max_grad_norm': 1.003176802748965,
        'vf_coef': 0.7380914329989935
    }
    def optimize_ppo(trial):
        return {
            'n_steps': trial.suggest_int('n_steps', 16, 2048),
            'gamma': trial.suggest_float('gamma', 0.8, 0.9999, log=True),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1, log=True),
            'ent_coef': trial.suggest_float('ent_coef', 0.00000001, 0.1, log=True),
            'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
            'n_epochs': trial.suggest_int('n_epochs', 1, 10),
            'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 1.0),
            'max_grad_norm': trial.suggest_float('max_grad_norm', 0.3, 5),
            'vf_coef': trial.suggest_float('vf_coef', 0.1, 1)
        }"""

    def create_env(df, window_size, frame_bound):
        env2 = CustomForexEnv(df=df, window_size=window_size, frame_bound=frame_bound)
        return DummyVecEnv([lambda: Monitor(env2)])

    """def objective(trial):
        model_params = optimize_ppo(trial)
        env = create_env(df, window_size=12, frame_bound=(12,50))
        model = RecurrentPPO("MlpLstmPolicy", env, verbose=0, **model_params)
        model.learn(total_timesteps=10000)
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
        return mean_reward

    # Hyperparameter optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, n_jobs=1)

    best_params = study.best_params
    print("Best hyperparameters: ", best_params)"""

    # Split data into training and validation sets
    train_size = int(len(df) * 0.8)
    train_df = df[:train_size]
    val_df = df[train_size:]

    # Create training and validation environments
    train_env = create_env(train_df, window_size=12, frame_bound=(12,200))
    val_env = create_env(val_df, window_size=12, frame_bound=(12,200))

    # Train the model
    model = RecurrentPPO("MlpLstmPolicy", train_env, verbose=1)
    eval_callback = EvalCallback(val_env, best_model_save_path='./logs/',
                                 log_path='./logs/', eval_freq=10000,
                                 deterministic=True, render=False)
    model.learn(total_timesteps=650000, callback=eval_callback)
    model.save("ppo_forex_model")
    return model, train_env

# Main trading function
def main_trading_loop():
    # Configure logging
    logging.basicConfig(filename='trading_log.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Log the start of the trading loop
    logging.info("Starting the main trading loop")

    symbol = "XAUUSD"
    timeframe = mt5.TIMEFRAME_D1
    start = datetime.datetime(2024, 1, 1)
    end = datetime.datetime.now()

    try:
        df = get_preprocessed_data(symbol, timeframe, start, end)
        logging.info("Data fetched and preprocessed successfully")
    except ValueError as e:
        logging.error(f"Error fetching data: {e}")
        return

    try:
        model, env = train_model(df)
        logging.info("Model trained successfully")
    except Exception as e:
        logging.error(f"Error training model: {e}")
        return
    
    # Evaluate the model
    obs = env.reset()
    while True: 
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            print("info", info)
            break

    # Log the end of the trading loop
    logging.info("Main trading loop finished")


if __name__ == "__main__":
    if not mt5.initialize(login=100003703, server="FBS-Demo", password="jV9}8d,)"):
        print("Failed to initialize MetaTrader5")
        mt5.shutdown()
        quit()
    
    main_trading_loop()
    mt5.shutdown()