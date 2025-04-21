import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import time
import pickle
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
import os
import time
from indicators1 import create_features
from indicators1 import classify_market_conditions, detect_market_regimes
from forex_env import TradingEnv
import datetime
from stable_baselines3.common.vec_env import DummyVecEnv
from rl import ActorCritic
import os
import numpy as np
import warnings
import threading
import pandas_ta as pta
from forex_env import Positions  # Import the Positions class from forex_env.py
warnings.filterwarnings('ignore')

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class TradingBot:
    def __init__(self, symbol, timeframe):
        self.symbol = symbol
        self.timeframe = timeframe
        self.df = pd.DataFrame()
        #print(self.df.head())
        self.window_size = 14
        self.current_position = Positions.Short  # Default to Short position


    def fetch_data(self):
        self.end = datetime.datetime.now() + datetime.timedelta(hours=4)
        self.start = self.end - datetime.timedelta(hours=5)
        rates = mt5.copy_rates_range(self.symbol, self.timeframe, self.start, self.end)

        if rates is None or len(rates) == 0:
            print(f"No data found for {self.symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df['volume'] = df['tick_volume']
        df.drop(['real_volume', 'tick_volume'], axis=1, inplace=True)

        if not df.empty:
            self.df = df.iloc[:-1]  # Ignore the last (incomplete) candle
        return self.df




    # Place a buy trade with only the entry price
    def place_buy_trade(self, entry_price, lotSize):
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            print(f"Symbol {self.symbol} not found.")
            return None

        # Place only one trade
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lotSize,
            "type": mt5.ORDER_TYPE_BUY,
            "price": entry_price,
            "deviation": 20,
            "magic": 234000,  # Unique magic number for the trade
            "comment": "Buy trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Failed to place buy order: {result.comment}")
            return None
        else:
            print(f"Buy order placed successfully with ticket: {result.order}")
            return result.order  # Return the order ID


    def place_sell_trade(self, entry_price, lotSize):
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            print(f"Symbol {self.symbol} not found.")
            return None

        # Place only one trade
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lotSize,
            "type": mt5.ORDER_TYPE_SELL,
            "price": entry_price,
            "deviation": 20,
            "magic": 234000,  # Unique magic number for the trade
            "comment": "Sell trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Failed to place sell order: {result.comment}")
            return None
        else:
            print(f"Sell order placed successfully with ticket: {result.order}")
            return result.order  # Return the order ID



    def monitor_and_update_sl(self):
        """Monitors the trade and updates the SL using Chandelier Exit, ensuring SL moves only in the trade's favor."""
        
        print("Waiting 1 minute before first SL adjustment...")
        time.sleep(60)  # Initial wait before adjusting SL
        self.wait_for_next_execution()  # Wait for the next 1-minute interval
        while True:
            self.fetch_data()  # ✅ Update df with new candles before SL calculation
            
            active_trade = self.get_active_trade()
            if not active_trade:
                print("No active trade found. Exiting SL monitoring.")
                break

            current_price = active_trade.price_current
            take_profit = active_trade.tp
            max_sl = active_trade.sl

            # Compute Chandelier Exit dynamically based on updated data
            chandelier_long, chandelier_short = self._calculate_chandelier_exit()

            trade_type = 'buy' if active_trade.type == mt5.ORDER_TYPE_BUY else 'sell'
            if trade_type == 'buy':
                new_sl = max(max_sl, chandelier_long)  # SL should only move up
            else:
                new_sl = min(max_sl, chandelier_short)  # SL should only move down

            if new_sl != max_sl:  # Only send update if SL actually changed
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": active_trade.ticket,
                    "sl": new_sl,
                    "tp": take_profit,
                }
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"Chandelier Exit SL updated to {new_sl} based on price {current_price}")
                    max_sl = new_sl  # Update the highest/lowest SL level
                else:
                    print(f"Failed to update SL: {result.comment}")

            time.sleep(60)  # ✅ Fetch data and update SL every 1 minutes




    def update_trade(self, trade_id):
        """Ensures the trade's SL is updated dynamically using Chandelier Exit."""
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            print(f"Symbol {self.symbol} not found.")
            return False

        positions = mt5.positions_get(symbol=self.symbol)
        if not positions:
            print("No active trades to update.")
            return False

        trade = next((p for p in positions if p.ticket == trade_id), None)
        if not trade:
            print("Could not find the trade.")
            return False

        entry_price = trade.price_open
        trade_type = 'buy' if trade.type == mt5.ORDER_TYPE_BUY else 'sell'
        atr = self._calculate_atr()

        # Compute Chandelier Exit values
        chandelier_long, chandelier_short = self._calculate_chandelier_exit()
        moving_average = self.df['close'][self.window_size:].mean()

        if trade_type == 'buy':
            stop_loss = chandelier_long
            take_profit = moving_average if moving_average > entry_price else entry_price + 0.25 * atr
            
            print(f"SL: {stop_loss}, TP: {take_profit}")
        else:
            stop_loss = chandelier_short
            take_profit = moving_average if moving_average < entry_price else entry_price - 0.25 * atr
            print(f"SL: {stop_loss}, TP: {take_profit}")

        # Apply Chandelier Exit SL update
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": trade_id,
            "sl": stop_loss,
            "tp": take_profit,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Failed to update trade: {result.comment}")
            return False

        # Start monitoring SL in a separate thread, ensuring it only moves in the correct direction
        monitor_thread = threading.Thread(target=self.monitor_and_update_sl)
        monitor_thread.start()

        return True



    def _calculate_chandelier_exit(self, atr_multiplier=2):
        lookback_period = 10
        _current_tick = len(self.df) - 1
        atr = self._calculate_atr()
        high_max = self.df['high'][_current_tick - lookback_period:_current_tick].max()
        low_min = self.df['low'][_current_tick - lookback_period:_current_tick].min()
        chandelier_long = high_max - (atr * atr_multiplier)
        chandelier_short = low_min + (atr * atr_multiplier)

        return chandelier_long, chandelier_short
    
    def _calculate_atr(self):
        # Simplified calculation of Average True Range (ATR)
        _current_tick = len(self.df) - 1
        high = self.df['high'][_current_tick - self.window_size:_current_tick].max()
        low = self.df['low'][_current_tick - self.window_size:_current_tick].min()
        close_prev = self.df['close'].iloc[max(_current_tick - 1, 0)]
        return max(high - low, abs(high - close_prev), abs(low - close_prev))
    
    def _calculate_rsi(self, window=14):
        _current_tick = len(self.df) - 1
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[_current_tick]

    def _is_favorable_condition(self):
        _current_tick = len(self.df) - 1
        #print(self.df.tail())
        #print(_current_tick)
        if self.df.empty or _current_tick < 0 or _current_tick >= len(self.df):
            print("Error: DataFrame is empty or _current_tick is out-of-bounds.")
            return False

        # Retrieve the current price and calculate indicators
        current_price = self.df['close'].iloc[_current_tick]
        print(f"Current price: {current_price}")
        moving_average = self.df['close'][_current_tick - self.window_size:_current_tick].mean()
        rsi = self._calculate_rsi()
        print(f"RSI: {rsi}")
        stddev = self.df['close'][_current_tick - self.window_size:_current_tick].std()
        lower_band = moving_average - 2 * stddev
        print(f"Lower band: {lower_band}")
        upper_band = moving_average + 2 * stddev
        print(f"Upper band: {upper_band}")
        

        # Confirm positions and evaluate conditions
        if self.current_position == Positions.Short and current_price < lower_band and rsi < 30:
            return True  # Only buy if price is at lower band & RSI is oversold
        elif self.current_position == Positions.Long and current_price > upper_band and rsi > 70:
            return True  # Only sell if price is at upper band & RSI is overbought
        # Print _current tick close price
        print(f"Current price: {current_price}")
        print("No favorable conditions for the current position.")
        return False



    def load_model_and_scaler(self):
        model_path = "best_model.pth"
        scaler_path = "scaler.pkl"

        try:
            model = torch.load(model_path, map_location=torch.device("cpu"))
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")

        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load scaler from {scaler_path}: {e}")

        return model, scaler


    def has_active_trade(self):
        positions = mt5.positions_get(symbol=self.symbol)
        return len(positions) > 0

    def get_active_trade(self):
        positions = mt5.positions_get(symbol=self.symbol)
        if positions:
            return positions[0]  # Assuming only one trade at a time
        return None

    def wait_for_next_execution(self):
        max_wait_time = 60
        start_time = time.time()
        while True:
            now = datetime.datetime.now()
            minute = now.minute
            second = now.second
            if (minute in range(60)) and (second in range(10)):
                print(f"Starting the next trading cycle at {now.strftime('%H:%M:%S')} UTC")
                break
            if time.time() - start_time > max_wait_time:
                print("Warning: Wait time exceeded limit. Resetting execution cycle.")
                break
            
            time.sleep(1)


def main_trading_loop():
    bot = TradingBot('XAUUSD', mt5.TIMEFRAME_M1)
    model, scaler = bot.load_model_and_scaler()

    while True:
        active_trade = bot.get_active_trade()
        if active_trade:
            print(f"Existing trade detected. Monitoring SL updates...")
            bot.monitor_and_update_sl()
            while bot.has_active_trade():
                time.sleep(60)
            print("Trade has ended. Restarting trading loop...")
            continue

        print("No active trade detected. Proceeding with normal trading logic...")

        bot.wait_for_next_execution()
        
        # Ensure new data is retrieved before proceeding
        df = bot.fetch_data()
        print(len(df))
        #print(df.tail(5))
        if df.empty:
            print("Skipping this cycle due to no data.")
            time.sleep(60)
            continue
        
        bot.df = df  # Update only when new data is available

        latest_hour = bot.df['time'].dt.hour.iloc[-1]
        if latest_hour not in [13, 18]:
            print(f"Not trading hours (Current hour: {latest_hour}). Waiting for 1 minute...")
            time.sleep(60)
            continue

        if bot._is_favorable_condition() == True:
            print("Favorable trading condition detected. Proceeding with prediction...")
        else:
            print("No favorable trading condition. Waiting for 1 minute...")
            time.sleep(60)
            continue

        
        print(f"Trading hour detected ({latest_hour}). Proceeding with prediction...")
        bot.df = create_features(bot.df)
        bot.df = classify_market_conditions(bot.df)
        bot.df = detect_market_regimes(bot.df)
        
        feature_columns = bot.df.columns.difference(['time', 'low', 'high', 'open', 'close', 'volume', 'spread'])
        bot.df[feature_columns] = bot.df[feature_columns].replace([np.inf, -np.inf], np.nan)
        bot.df.fillna(method="ffill", inplace=True)
        bot.df.fillna(method="bfill", inplace=True)
        
        try:
            bot.df[feature_columns] = scaler.transform(bot.df[feature_columns])
        except ValueError:
            print("Error: Feature columns mismatch or missing data. Skipping this iteration.")
            time.sleep(60)
            continue
        
        print(bot.df.tail(5))
        print(len(bot.df))
        env = DummyVecEnv([lambda: TradingEnv(bot.df, window_size=14, frame_bound=(14, len(bot.df)))])
        obs, _ = env.envs[0].reset()
        action, _ = model.predict(obs)
        print(f"Predicted Action: {action}")
        
        chandelier_long, chandelier_short = bot._calculate_chandelier_exit()
        order_details = None
        
        if action == 1:
            entry_price = bot.df['close'].iloc[-1]
            print(entry_price)
            if chandelier_long > (entry_price - 1.35):
                print("Chandelier Exit is above the entry price. Waiting for 1 minutes...")
                time.sleep(60)
                continue
            if bot.current_position == Positions.Long:
                print("Already in a long position. Waiting for 1 minute...")
                time.sleep(60)
                continue
            lotSize = 0.01
            order_details = bot.place_buy_trade(entry_price, lotSize)
            bot.current_position = Positions.Long
        
        elif action == 0:
            entry_price = bot.df['close'].iloc[-1]
            print(entry_price)
            if chandelier_short < (entry_price + 1.35):
                print("Chandelier Exit is below the entry price. Waiting for 1 minute...")
                time.sleep(60)
                continue
            if bot.current_position == Positions.Short:
                print("Already in a short position. Waiting for 1 minute...")
                time.sleep(60)
                continue
            lotSize = 0.01
            order_details = bot.place_sell_trade(entry_price, lotSize)
            bot.current_position = Positions.Short
        
        elif action == 2:
            print(f"Current close price: {bot.df['close'].iloc[-1]}")
            print("Holding position. Waiting for 1 minute...")
            time.sleep(60)
            continue
        
        if order_details:
            trade_ids = order_details
            if trade_ids:
                bot.update_trade(trade_ids)
                while bot.has_active_trade():
                    time.sleep(60)
                print("Trade has ended. Restarting trading loop...")
                continue
        else:
            print("No favorable trading condition. Waiting for 1 minute...")
            time.sleep(60)
            continue

if __name__ == "__main__":
    if not mt5.initialize(login=100003703, server="FBS-Demo", password="jCv0+4N."):
        print("Failed to initialize MetaTrader5")
        mt5.shutdown()
        quit()
    main_trading_loop()
    mt5.shutdown()
