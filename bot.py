import MetaTrader5 as mt5
import os
import time
import datetime
import numpy as np
import warnings
import threading
import pandas as pd
import json

with open("config.json") as f:
    config = json.load(f)

warnings.filterwarnings('ignore')

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class TradingBot:
    def __init__(self):
        self.tp1_closed = False
        self.symbol = "XAUUSD"
        self.days = 1
        self.df_15 = pd.DataFrame()
        self.df_5 = pd.DataFrame()
        self.df_h1 = pd.DataFrame()
        self.df_h4 = pd.DataFrame()

    def sync_time(self):
        """Sync system time with MT5 server time."""
        mt5_time = datetime.datetime.fromtimestamp(mt5.symbol_info_tick(self.symbol).time)  # Get time from MT5 server
        system_time = datetime.datetime.now()  # System time (UTC)
        
        # Check if the times match
        if mt5_time != system_time:
            print(f"Warning: Time mismatch detected! Adjusting for time difference...")
            time_difference = mt5_time - system_time
            adjusted_time = system_time + time_difference
            return adjusted_time
        return system_time  # No adjustment needed

    def fetch_data_5(self):
        adjusted_time = self.sync_time()  # Sync time with MT5 server time
        end = adjusted_time
        start = end - datetime.timedelta(days=self.days)
        symbol = self.symbol  # Symbol to fetch data for
        timeframe = mt5.TIMEFRAME_M5
        rates = mt5.copy_rates_range(symbol, timeframe, start, end)

        if rates is None or len(rates) == 0:
            print(f"No data found for {self.symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df['candle_range_size'] = abs(df['high'] - df['low'])

        # Check if the last candle is incomplete
        last_candle_time = df['time'].iloc[-1]

        # Round adjusted_time to the nearest minute
        now_time = datetime.datetime.now().replace(second=0, microsecond=0)

        # If the last candle time is greater than or equal to the current time, drop the last candle
        if last_candle_time >= now_time:
            df = df.iloc[:-1]  # Drop the incomplete candle

        return df

    
    def rsi(self, df_5, period=14):
        delta = df_5['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def calculate_atr(self, df, period=14):
        df_copy = df.copy()
        df_copy['H-L'] = df_copy['high'] - df_copy['low']
        df_copy['H-C'] = abs(df_copy['high'] - df_copy['close'].shift())
        df_copy['L-C'] = abs(df_copy['low'] - df_copy['close'].shift())
        df_copy['TR'] = df_copy[['H-L', 'H-C', 'L-C']].max(axis=1)
        df_copy['ATR'] = df_copy['TR'].rolling(window=period).mean()
        return df_copy['ATR']
    
    def candles_patern_5(self, df_5):
        # Define the minimum and maximum allowable range for candles
        min_range = 2.00  # Adjust based on your market (e.g., 0.001 for Forex)

        # Calculate the range of the two most recent candles
        candle_1 = df_5.iloc[-1]
        candle_2 = df_5.iloc[-2]
        candle_3 = df_5.iloc[-3]

        def is_valid_range(candle):
            # use absolute range for the candle
            range_size = abs(candle['high'] - candle['low'])
            return range_size > min_range

        # Check if recent candles meet the range criteria
        candles = (
            #candle_1['close'] > candle_1['open'] and
            is_valid_range(candle_1) and
            is_valid_range(candle_2) and
            is_valid_range(candle_3)
        )

        return candles
    
    def fetch_data_15(self):
        adjusted_time = self.sync_time()  # Get the synchronized time
        end = adjusted_time
        start = end - datetime.timedelta(days=self.days)  # Set the start time for the data fetch
        symbol = self.symbol  # Symbol to fetch data for
        timeframe = mt5.TIMEFRAME_M15
        rates = mt5.copy_rates_range(symbol, timeframe, start, end)  # Fetch the M15 data

        if rates is None or len(rates) == 0:
            print(f"No data found for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        now_time = datetime.datetime.now().replace(second=0, microsecond=0)  # Get the current time rounded to the nearest minute
        
        # If adjusted_time is a multiple of 15 minutes, check if we need to drop the last incomplete candle
        last_candle_time = df['time'].iloc[-1]
        if last_candle_time >= now_time:
            df = df.iloc[:-1]  # Drop the incomplete candle if the last candle is not complete
        
        return df

        

    
    def ema(self, df_15, period=5):

        ema_val = df_15['close'].ewm(span=period, adjust=False).mean().iloc[-1]
        bullish_ema = df_15['close'].iloc[-1] > ema_val
        bearish_ema = df_15['close'].iloc[-1] < ema_val

        return bullish_ema, bearish_ema
    
    def ema_long(self, df_15):
        ema_long = df_15['close'].ewm(span=20).mean().iloc[-1]
        uptrend = df_15['close'].iloc[-1] > ema_long
        downtrend = df_15['close'].iloc[-1] < ema_long

        return uptrend, downtrend
    
    def fetch_data_h1(self):
        adjusted_time = self.sync_time()
        end = adjusted_time
        start = end - datetime.timedelta(days=2)
        symbol = self.symbol
        timeframe = mt5.TIMEFRAME_H1
        rates = mt5.copy_rates_range(symbol, timeframe, start, end)
        if rates is None or len(rates) == 0:
            print(f"No data found for {symbol}")
            return pd.DataFrame()
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        now_time = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)
        # Check if the last candle is incomplete
        last_candle_time = df['time'].iloc[-1]
        if last_candle_time >= now_time:
            df = df.iloc[:-1]
        return df
    
    def fetch_data_h4(self):
        adjusted_time = self.sync_time()
        end = adjusted_time
        start = end - datetime.timedelta(days=6)
        symbol = self.symbol
        timeframe = mt5.TIMEFRAME_H4
        rates = mt5.copy_rates_range(symbol, timeframe, start, end)
        if rates is None or len(rates) == 0:
            print(f"No data found for {symbol}")
            return pd.DataFrame()
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        now_time = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)
        # Check if the last candle is incomplete
        last_candle_time = df['time'].iloc[-1]
        if last_candle_time >= now_time:
            df = df.iloc[:-1]
        return df
    
    def macro_trend(self, df_h1, df_h4):
        h1_ema = df_h1['close'].ewm(span=20, adjust=False).mean()
        h4_ema = df_h4['close'].ewm(span=20, adjust=False).mean()

        h1_trend = "UP" if df_h1['close'].iloc[-1] > h1_ema.iloc[-1] else "DOWN"
        h4_trend = "UP" if df_h4['close'].iloc[-1] > h4_ema.iloc[-1] else "DOWN"

        return h1_trend, h4_trend


    def place_buy_trade(self, entry_price, lotSize, tp, sl):
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            print(f"Symbol XAUUSD not found.")
            return None
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lotSize,
            "type": mt5.ORDER_TYPE_BUY,
            "price": entry_price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 234000,
            "comment": f"Buy Trade TP {tp}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        # Validate the trade request parameters
        if not mt5.symbol_info_tick(self.symbol):
            print(f"Error: Unable to retrieve tick data for {self.symbol}. Check if the symbol is available.")
            return None

        result = mt5.order_send(request)
        if result is None:
            print(f"Error: mt5.order_send returned None. Request: {request}")
            return None
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Failed to place buy order: {result.comment}")
            return None

        return result.order

    def place_sell_trade(self, entry_price, lotSize, tp, sl):
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            print(f"Symbol XAUUSD not found.")
            return None
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lotSize,
            "type": mt5.ORDER_TYPE_SELL,
            "price": entry_price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 234000,
            "comment": f"SELL trade TP {tp}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        # Validate the trade request parameters
        if not mt5.symbol_info_tick(self.symbol):
            print(f"Error: Unable to retrieve tick data for {self.symbol}. Check if the symbol is available.")
            return None

        result = mt5.order_send(request)
        if result is None:
            print(f"Error: mt5.order_send returned None. Request: {request}")
            return None
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Failed to place sell order: {result.comment}")
            return None

        return result.order


    def has_active_trade(self):
        positions = mt5.positions_get(symbol="XAUUSD")
        return positions is not None and len(positions) > 0

    def get_active_trades(self):
        positions = mt5.positions_get(symbol="XAUUSD")
        return positions if positions else []

    def wait_for_next_execution(self):
        while True:
            now = datetime.datetime.now()
            if now.minute % 5 == 0 and now.second < 10:
                print(f"Starting trading cycle at {now.strftime('%H:%M:%S')} UTC")
                break
            time.sleep(1)


def main_trading_loop():
    bot = TradingBot()
    lotSize = 0.01
    TP_OFFSET = 1.1
    SL_OFFSET = 1.0
    valid_hours = [15, 16, 17, 18]

    while True:
        active_trade = bot.has_active_trade()
        if active_trade:
            while bot.has_active_trade():
                print("Active trade detected. Waiting for trade to close...")
                time.sleep(30)
            print("Trade has ended")
            continue
        print("No active trade detected. Proceeding with normal trading logic...")
        bot.wait_for_next_execution()
        bot.df_5 = bot.fetch_data_5()
        print(bot.df_5.tail(5))

        # chenck if current time is a multiple of 15 minutes
        now = datetime.datetime.now()
    
        if now.minute % 15 == 0:
            bot.df_15 = bot.fetch_data_15()
        elif bot.df_15.empty and now.minute % 15 != 0:
            df = bot.fetch_data_15()
            # remove last candle
            bot.df_15 = df.iloc[:-1]
        print(bot.df_15.tail(5))

        if now.minute == 0:
            bot.df_h1 = bot.fetch_data_h1()
        elif bot.df_h1.empty and now.minute != 0:
            df = bot.fetch_data_h1()
            # remove last candle
            bot.df_h1 = df

        print(bot.df_h1.tail(5))
        if now.hour % 4 == 0 and now.minute == 0:
            bot.df_h4 = bot.fetch_data_h4()
        elif bot.df_h4.empty and now.hour % 4 != 0 and now.minute != 0:
            df = bot.fetch_data_h4()
            # remove last candle
            bot.df_h4 = df.iloc[:-1]
        print(bot.df_h4.tail(5))
        


        if bot.df_5.empty or bot.df_15.empty or bot.df_h1.empty or bot.df_h4.empty:
            print("Data retrieval failed. Retrying after 5 minutes...")
            time.sleep(300)
            continue
        candles = bot.candles_patern_5(bot.df_5)
        print("Candles_range:", candles)
        bullish_ema, bearish_ema = bot.ema(bot.df_15, period=5)
        print("Bullish EMA:", bullish_ema)
        print("Bearish EMA:", bearish_ema)
        uptrend, downtrend = bot.ema_long(bot.df_15)
        print("Uptrend:", uptrend)
        print("Downtrend:", downtrend)
        h1_trend, h4_trend = bot.macro_trend(bot.df_h1, bot.df_h4)
        print("H1 Trend:", h1_trend)
        print("H4 Trend:", h4_trend)
        rsi = bot.rsi(bot.df_5, period=14)
        atr_series = bot.calculate_atr(bot.df_5, period=14)
        atr_tp = atr_series.iloc[-1] * TP_OFFSET
        atr_sl = atr_series.iloc[-1] * SL_OFFSET
        print("ATR TP:", atr_tp)
        print("ATR SL:", atr_sl)
        print("RSI:", rsi)

        if bot.df_5['time'].iloc[-1].hour in valid_hours:
            if candles and bullish_ema and uptrend and rsi < 50 and h1_trend == "UP" and h4_trend == "UP":
                action = 1

            elif candles and bearish_ema and downtrend and rsi > 50 and h1_trend == "DOWN" and h4_trend == "DOWN":
                action = 2
            else:
                action = 0
        else:
            action = 0

        print("Action:", action)
        if action == 1:
            entry_price = bot.df_5['close'].iloc[-1]
            print("Entry price:", entry_price)
            # rounding the sl and tps to 2 decimal places
            tp = float(f"{entry_price + atr_tp:.2f}")
            sl = float(f"{entry_price - atr_sl:.2f}")
            print(f"SL: {sl}, TP1: {tp}")
            order_details = bot.place_buy_trade(entry_price, lotSize, tp, sl)

        elif action == 2:
            entry_price = bot.df_5['close'].iloc[-1]
            print("Entry price:", entry_price)
            # rounding the sl and tps to 2 decimal places
            tp = float(f"{entry_price - atr_tp:.2f}")
            sl = float(f"{entry_price + atr_sl:.2f}")
            # rounding the sl and tps to 2 decimal places
            print(f"SL: {sl}, TP: {tp}")
            order_details = bot.place_sell_trade(entry_price, lotSize, tp, sl)

        elif action == 0:
            print("No trade signal detected. Waiting for the next cycle.")
            time.sleep(300)
            continue

        if order_details:
            trade_ids = order_details
            if trade_ids:
                while bot.has_active_trade():
                    now = datetime.datetime.now()
                    if now.minute % 15 == 0:
                        bot.df_15 = bot.fetch_data_15()
                    if now.minute == 0:
                        bot.df_h1 = bot.fetch_data_h1()
                    if now.hour % 4 == 0 and now.minute == 0:
                        bot.df_h4 = bot.fetch_data_h4()
                    time.sleep(30)
                print("Trade has ended. Restarting trading loop...")
                continue
        else:
            print("Failed to place trade. Retrying in 5 minutes...")
            time.sleep(490)
            continue

if __name__ == "__main__":
    if not mt5.initialize(login=config["login"], password=config["password"], server=config["server"]):
        print(f"Failed to initialize MetaTrader5, error code: {mt5.last_error()}")
        mt5.shutdown()
        quit()
    main_trading_loop()
    mt5.shutdown()
