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
from main1 import ActorCritic
import datetime
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import numpy as np
import warnings
import threading
import pandas_ta as pta
warnings.filterwarnings('ignore')

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import logging
import sys

logger = logging.getLogger("trading_bot1")  # Use a unique name
logger.setLevel(logging.DEBUG)

if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)


class TradingBot:
    def __init__(self, symbol, timeframe):
        self.symbol = symbol
        self.timeframe = timeframe
        self.df = pd.DataFrame()
        #print(self.df.head())
        self.window_size = 14
        self.current_position = 'Short' # Initial position state


    def fetch_data(self):
        self.end = datetime.datetime.now() + datetime.timedelta(hours=4)
        self.start = self.end - datetime.timedelta(days=42)
        rates = mt5.copy_rates_range(self.symbol, self.timeframe, self.start, self.end)

        if rates is None or len(rates) == 0:
            logger.error(f"No data found for {self.symbol}")
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
            logger.error(f"Failed to place buy order: {result.comment}")
            return None
        else:
            logger.info(f"Buy order placed successfully with ticket: {result.order}")
            return result.order  # Return the order ID


    def place_sell_trade(self, entry_price, lotSize):
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            logger.error(f"Symbol {self.symbol} not found.")
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
            logger.error(f"Failed to place sell order: {result.comment}")
            return None
        else:
            logger.info(f"Sell order placed successfully with ticket: {result.order}")
            return result.order  # Return the order ID



    def monitor_and_update_sl(self):
        """Monitors the trade and updates the SL using Chandelier Exit, ensuring SL moves only in the trade's favor."""
        
        logger.info("Waiting 30 minutes before first SL adjustment...")
        time.sleep(60)  # Initial wait before adjusting SL
        self.wait_for_next_execution()  # Wait for the next 30-minute interval
        while True:
            self.fetch_data()  # ✅ Update df with new candles before SL calculation
            
            active_trade = self.get_active_trade()
            max_sl = None  # Initialize max_sl to avoid UnboundLocalError
            if not active_trade:
                logger.info("No active trade found. Exiting SL monitoring.")
                break

            current_price = active_trade.price_current
            take_profit = active_trade.tp

            # Compute Chandelier Exit dynamically based on updated data
            chandelier_long, chandelier_short = self._calculate_chandelier_exit()

            trade_type = 'buy' if active_trade.type == mt5.ORDER_TYPE_BUY else 'sell'
            if trade_type == 'buy':
                # SL should move to break-even if price is above (entry price + 1.5 * ATR)
                if current_price > active_trade.price_open + (1.5 * self._calculate_atr()):  # Break-even point
                    max_sl = active_trade.price_open

            else:
                # SL should move to break-even if price is below (entry price - 1.5 * ATR)
                if current_price < active_trade.price_open - (1.5 * self._calculate_atr()):
                    max_sl = active_trade.price_open

            if max_sl is not None:
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": active_trade.ticket,
                    "sl": max_sl,
                    "tp": take_profit,
                }
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"SL updated successfully to {max_sl}")
                    # stop running the loop if sl is updated
                    break
                else:
                    logger.debug(f"Failed to update SL: {result.comment}")

            time.sleep(1800)  # ✅ Fetch data and update SL every 30 minutes




    def update_trade(self, trade_id):
        """Ensures the trade's SL is updated dynamically using Chandelier Exit."""
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            logger.error(f"Symbol {self.symbol} not found.")
            return False

        positions = mt5.positions_get(symbol=self.symbol)
        if not positions:
            logger.error("No active trades to update.")
            return False

        trade = next((p for p in positions if p.ticket == trade_id), None)
        if not trade:
            logger.error("Could not find the trade.")
            return False

        entry_price = trade.price_open
        trade_type = 'buy' if trade.type == mt5.ORDER_TYPE_BUY else 'sell'
        atr = self._calculate_atr()

        if trade_type == 'buy':
            stop_loss = entry_price - (1.5 * atr)  # Initial SL at 1.5x ATR
            take_profit = entry_price + (2 * atr)  # Use 2x ATR as TP
            logger.info(f"SL: {stop_loss}, TP: {take_profit}")
        else:
            stop_loss = entry_price + (1.5 * atr)  # Initial SL at 1.5x ATR
            take_profit = entry_price - (2 * atr)  # Use 2x ATR as TP
            logger.info(f"SL: {stop_loss}, TP: {take_profit}")

        # Apply Chandelier Exit SL update
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": trade_id,
            "sl": stop_loss,
            "tp": take_profit,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to update trade: {result.comment}")
            return False

        # Start monitoring SL in a separate thread, ensuring it only moves in the correct direction
        monitor_thread = threading.Thread(target=self.monitor_and_update_sl)
        monitor_thread.start()

        return True



    def _calculate_chandelier_exit(self, atr_multiplier=0.5):
        lookback_period = self.window_size
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

    def _is_favorable_condition(self, action):
        _current_tick = len(self.df) - 1
        self.ema_5 = self.df['close'].ewm(span=5, adjust=False).mean()
        self.ema_10 = self.df['close'].ewm(span=10, adjust=False).mean()
        ema_5 = self.ema_5.iloc[_current_tick]
        ema_10 = self.ema_10.iloc[_current_tick]
        
        if action == 1 and ema_5 > ema_10:
            return True
        elif action == 0 and ema_5 < ema_10:
            return True
        return False



    def load_model(self):
        model_path = "best_model.pth"

        try:
            model = torch.load("best_model.pth", map_location=torch.device("cpu"))  # ✅ Load full model
            model.eval()  # Set to evaluation mode
        except Exception as e:
                raise RuntimeError(f"Failed to load model from {model_path}: {e}")

        return model


    def has_active_trade(self):
        positions = mt5.positions_get(symbol=self.symbol)
        return len(positions) > 0

    def get_active_trade(self):
        positions = mt5.positions_get(symbol=self.symbol)
        if positions:
            return positions[0]  # Assuming only one trade at a time
        return None

    def wait_for_next_execution(self):
        max_wait_time = 1800  # 30 minutes
        start_time = time.time()
        while True:
            now = datetime.datetime.now()
            minute = now.minute
            second = now.second
            if (minute in [0, 50, 30]) and (second in range(2, 10)):
                logger.info(f"Starting the next trading cycle at {now.strftime('%H:%M:%S')} UTC")
                break
            if time.time() - start_time > max_wait_time:
                logger.warning("Warning: Wait time exceeded limit. Resetting execution cycle.")
                break
            
            time.sleep(2)


def main_trading_loop():
    bot = TradingBot('EURUSD', mt5.TIMEFRAME_M30)
    scaler = pickle.load(open("scaler.pkl", "rb"))

    while True:
        active_trade = bot.get_active_trade()
        if active_trade:
            logger.info(f"Existing trade detected. Monitoring SL updates...")
            bot.monitor_and_update_sl()
            while bot.has_active_trade():
                time.sleep(60)
            logger.info("Trade has ended. Restarting trading loop...")
            continue

        logger.info("No active trade detected. Proceeding with normal trading logic...")

        bot.wait_for_next_execution()
        
        # Ensure new data is retrieved before proceeding
        df = bot.fetch_data()
        logger.info(len(df))
        if df.empty:
            logger.info("Skipping this cycle due to no data.")
            time.sleep(1800)
            continue
        
        bot.df = df  # Update only when new data is available

        """latest_hour = bot.df['time'].dt.hour.iloc[-1]
        if latest_hour in [18, 19, 20, 21, 22, 23, 1, 0]:
            logger.info(f"Not trading hours (Current hour: {latest_hour}). Waiting for 30 minutes...")
            time.sleep(1800)
            continue
        
        logger.info(f"Trading hour detected ({latest_hour}). Proceeding with prediction...")"""
        bot.df = create_features(bot.df)
        
        feature_columns = bot.df.columns.difference(['time', 'close', 'low', 'high', 'open', 'volume', 'spread', 'hour_of_day', 'regime', 'market_condition', 'bullish_signal', 'bearish_signal'])
        bot.df[feature_columns] = bot.df[feature_columns].replace([np.inf, -np.inf], np.nan)
        bot.df.fillna(method="ffill", inplace=True)
        bot.df.fillna(method="bfill", inplace=True)
        
        try:
            bot.df[feature_columns] = scaler.transform(bot.df[feature_columns])
        except ValueError:
            logger.error("Error: Feature columns mismatch or missing data. Skipping this iteration.")
            time.sleep(1800)
            continue
        
        logger.info(bot.df.tail(5))
        logger.info(len(bot.df))
        env = TradingEnv(bot.df, window_size=14, frame_bound=(14, len(bot.df)))
        model = bot.load_model()

        reset_result = env.reset()
        obs, _ = reset_result if isinstance(reset_result, tuple) else (reset_result, None)
        
        # Predict action
        action, _ = model.predict(obs)

        if bot._is_favorable_condition(action):
            logger.info("Favorable trading condition detected.")
        else:
            logger.info("Unfavorable trading condition detected. Waiting for 30 minutes...")
            time.sleep(1800)
            continue

        logger.info(f"Predicted Action: {action}")
        
        order_details = None
        
        if action == 1:
            entry_price = bot.df['close'].iloc[-1]
            logger.info(entry_price)
            lotSize = 0.01
            order_details = bot.place_buy_trade(entry_price, lotSize)
        
        elif action == 0:
            entry_price = bot.df['close'].iloc[-1]
            logger.info(entry_price)
            lotSize = 0.01
            order_details = bot.place_sell_trade(entry_price, lotSize)
        
        elif action == 2:
            logger.info(f"Current close price: {bot.df['close'].iloc[-1]}")
            logger.info("Holding position. Waiting for 30 minutes...")
            time.sleep(1750)
            continue
        
        if order_details:
            trade_ids = order_details
            if trade_ids:
                bot.update_trade(trade_ids)
                while bot.has_active_trade():
                    time.sleep(60)
                logger.info("Trade has ended. Restarting trading loop...")
                continue
        else:
            logger.info("No favorable trading condition. Waiting for 30 minutes...")
            time.sleep(1750)
            continue

if __name__ == "__main__":
    if not mt5.initialize(login=100003703, server="FBS-Demo", password="jCv0+4N."):
        logger.error("Failed to initialize MetaTrader5")
        mt5.shutdown()
        quit()
    main_trading_loop()
    mt5.shutdown()
