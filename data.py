import MetaTrader5 as mt5
import pandas as pd
import datetime
import numpy as np


class TradingBotSimulator:
    def __init__(self):
        self.symbol = "XAUUSD"
        self.days = 10
        

    def fetch_data_5(self):
        end = datetime.datetime.now() + datetime.timedelta(hours=4)
        start = end - datetime.timedelta(days=self.days)
        symbol = "XAUUSD"
        timeframe = mt5.TIMEFRAME_M5
        rates = mt5.copy_rates_range(symbol, timeframe, start, end)

        if rates is None or len(rates) == 0:
            print(f"No data found for {self.symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df['candle_range_size'] = abs(df['high'] - df['low'])

        # mean, median, and std, min, max, mode of the candle range size
        #mean = df['candle_range_size'].mean()
        #median = df['candle_range_size'].median()
        #std = df['candle_range_size'].std()
        #min_value = df['candle_range_size'].min()
        #max_value = df['candle_range_size'].max()
        #mode = df['candle_range_size'].mode()[0]
        #print(f"Mean: {mean}, Median: {median}, Std: {std}, Min: {min_value}, Max: {max_value}, Mode: {mode}")

        return df
    
    def fetch_data_15(self):
        end = datetime.datetime.now() + datetime.timedelta(hours=4)
        start = end - datetime.timedelta(days=self.days)
        symbol = "XAUUSD"
        timeframe = mt5.TIMEFRAME_M15
        rates = mt5.copy_rates_range(symbol, timeframe, start, end)

        if rates is None or len(rates) == 0:
            print(f"No data found for {self.symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')


        return df
    
    def fetch_data_h1(self):
        end = datetime.datetime.now() + datetime.timedelta(hours=4)
        start = end - datetime.timedelta(days=self.days)
        symbol = "XAUUSD"
        timeframe = mt5.TIMEFRAME_H1
        rates = mt5.copy_rates_range(symbol, timeframe, start, end)

        if rates is None or len(rates) == 0:
            print(f"No data found for {self.symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        return df
    
    def fetch_data_h4(self):
        end = datetime.datetime.now() + datetime.timedelta(hours=4)
        start = end - datetime.timedelta(days=self.days)
        symbol = "XAUUSD"
        timeframe = mt5.TIMEFRAME_H4
        rates = mt5.copy_rates_range(symbol, timeframe, start, end)

        if rates is None or len(rates) == 0:
            print(f"No data found for {self.symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        return df
    
def main():

    # Create an instance of the TradingBotSimulator
    bot = TradingBotSimulator()

    # Fetch and save data
    df_5 = bot.fetch_data_5()
    df_5.to_csv("XAUUSD_M5.csv", index=False)
    df_15 = bot.fetch_data_15()
    df_15.to_csv("XAUUSD_M15.csv", index=False)
    df_h1 = bot.fetch_data_h1()
    df_h1.to_csv("XAUUSD_H1.csv", index=False)
    df_h4 = bot.fetch_data_h4()
    df_h4.to_csv("XAUUSD_H4.csv", index=False)

if __name__ == "__main__":
    if not mt5.initialize(login=102167582, server="FBS-Demo", password=".{:I2Nn("):
        print("Failed to initialize MetaTrader5")
        mt5.shutdown()
        quit()
    main()
    mt5.shutdown()