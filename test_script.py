import MetaTrader5 as mt5
import numpy as np
import datetime
import pandas as pd
import gymnasium as gym
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
from main import CustomForexEnv
from gymnasium.wrappers import TimeLimit
import time

# Custom environment
class CustomForexEnv(ForexEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # Override _process_data to add signals
    def _process_data(self):
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]
        prices = self.df.loc[:, "Low"].to_numpy()[start:end]
        signal_features = self.df.loc[:, [
            "Low", "Volume", "SMA_12", "EMA_50", "RSI", "MACD", "Signal_Line",
            "OBV", "Upper_BB", "Middle_BB", "Lower_BB", "sma_100", "sma_200", 
            "ATR", "rolling_std", "slowk", "slowd", "ROC", "CCI", "Williams_%R", 
            "MFI", "keltner_upper", "keltner_lower", "price_rsi_divergence", 
            "ha_open", "ha_high", "ha_low", "ha_close", "fisher_transform", 
            "vwap", "eri_bull_power", "eri_bear_power", "cmo", "kvo", "dpo", 
            "mfi_divergence"
        ]].to_numpy()[start:end]
        return prices, signal_features

# TradingBot class to encapsulate all trading logic
class TradingBot:
    def __init__(self, symbol, timeframe, num_bars):
        self.symbol = symbol
        self.timeframe = timeframe
        self.num_bars = num_bars
        self.modelA, self.scalerA = self.load_modelA_and_scalerA()


    # Function to fetch historical data from MetaTrader5
    def fetch_data(symbol, timeframe, start, end, output_file="trading_data.csv"):
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

    # Place a buy trade
    def place_buy_trade(self, entry_price, stop_loss, take_profit, lotSize):

        # Place the initial buy order with the first take profit level
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lotSize,  # Divide the lot size into three parts
            "type": mt5.ORDER_TYPE_BUY,
            "price": entry_price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 20,
            "magic": 234000,
            "comment": "Buy trade TP1",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Failed to place buy order: {result.comment}")
            return None

        return [result.order]

    # Place a sell trade
    def place_sell_trade(self, entry_price, stop_loss, take_profit, lotSize):

        # Place the initial sell order with the first take profit level
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lotSize,  # Divide the lot size into three parts
            "type": mt5.ORDER_TYPE_SELL,
            "price": entry_price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 20,
            "magic": 234000,
            "comment": "Sell trade TP1",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Failed to place sell order: {result.comment}")
            return None

        return [result.order]

    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        df.set_index("Date", inplace=True)
        df.drop(columns=["Unnamed: 0"], inplace=True)
        df['Volume'] = df['Volume'].apply(lambda x: float(str(x).replace(",", "")))
        df = self.create_features(df)
        df.fillna(0, inplace=True)
        return df

    # Check if any active trades are open
    def has_active_trade(self):
        positions = mt5.positions_get(symbol=self.symbol)
        return len(positions) > 0

    def wait_for_next_execution(self):
        while True:
            now = datetime.datetime.now()
            minute = now.minute
            second = now.second
            if (1 < now.hour < 24) and (minute in [0]) and (second in range(20)):
                print(f"Starting the next trading cycle at {now.strftime('%H:%M:%S')} UTC")
                break
            else:
                time.sleep(40)
