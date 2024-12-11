import MetaTrader5 as mt5
import numpy as np
import pandas as pd

# Helper functions for indicators
def compute_dpo(Close, period=20):
    sma = Close.shift(int((period / 2) + 1)).rolling(window=period).mean()
    dpo = Close - sma
    return dpo
def compute_kvo(Close, High, Low, Volume, short_period=34, long_period=55):
    trend = ((High + Low + Close) / 3) - ((High.shift(1) + Low.shift(1) + Close.shift(1)) / 3)
    volume_force = trend * Volume
    short_kvo = volume_force.ewm(span=short_period, adjust=False).mean()
    long_kvo = volume_force.ewm(span=long_period, adjust=False).mean()
    kvo = short_kvo - long_kvo
    return kvo
def detect_mfi_divergence(Close, mfi, lookback=5):
    price_highs = Close.rolling(window=lookback).max()
    price_lows = Close.rolling(window=lookback).min()
    mfi_highs = mfi.rolling(window=lookback).max()
    mfi_lows = mfi.rolling(window=lookback).min()
    divergence = np.where((Close > price_highs.shift(1)) & (mfi < mfi_highs.shift(1)), 1,
                          np.where((Close < price_lows.shift(1)) & (mfi > mfi_lows.shift(1)), -1, 0))
    return pd.Series(divergence, index=Close.index)

def compute_cmo(Close, period=14):
    delta = Close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).sum()
    loss = -delta.where(delta < 0, 0).rolling(window=period).sum()
    cmo = 100 * (gain - loss) / (gain + loss)
    return cmo

def compute_fisher_transform(Close, period=10):
    min_low = Close.rolling(window=period).min()
    max_high = Close.rolling(window=period).max()
    epsilon = 1e-10  # Small value to avoid division by zero
    value = 2 * ((Close - min_low) / (max_high - min_low + epsilon) - 0.5)
    value = np.clip(value, -0.999, 0.999)  # Clip values to avoid log of zero
    fisher_transform = (np.log((1 + value) / (1 - value + epsilon))).rolling(window=2).mean()
    return fisher_transform

def compute_vwap(Close, Volume):
    cum_volume_price = (Close * Volume).cumsum()
    cum_volume = Volume.cumsum()
    vwap = cum_volume_price / cum_volume
    return vwap

def compute_eri(High, Low, Close, ema_window=13):
    ema = Close.ewm(span=ema_window, adjust=False).mean()
    bull_power = High - ema
    bear_power = Low - ema
    return bull_power, bear_power

def compute_keltner_channels(Close, High, Low, ema_window=20, atr_window=10, atr_multiplier=2):
    ema = Close.ewm(span=ema_window, adjust=False).mean()
    atr = compute_atr(High, Low, Close, window=atr_window)
    upper_band = ema + (atr_multiplier * atr)
    lower_band = ema - (atr_multiplier * atr)
    return upper_band, lower_band

def detect_divergence(Close, indicator, lookback=5):
    price_highs = Close.rolling(window=lookback).max()
    price_lows = Close.rolling(window=lookback).min()
    indicator_highs = indicator.rolling(window=lookback).max()
    indicator_lows = indicator.rolling(window=lookback).min()
    divergence = np.where((Close > price_highs.shift(1)) & (indicator < indicator_highs.shift(1)), 1,
                          np.where((Close < price_lows.shift(1)) & (indicator > indicator_lows.shift(1)), -1, 0))
    return pd.Series(divergence, index=Close.index)

def compute_heikin_ashi(df):
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_open = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
    ha_high = pd.concat([df['High'], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([df['Low'], ha_open, ha_close], axis=1).min(axis=1)
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

def compute_atr(High, Low, Close, window=14):
    tr1 = High - Low
    tr2 = np.abs(High - Close.shift(1))
    tr3 = np.abs(Low - Close.shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

def compute_stochastic(High, Low, Close, window=14):
    lowest_low = Low.rolling(window=window).min()
    highest_high = High.rolling(window=window).max()
    slowk = 100 * (Close - lowest_low) / (highest_high - lowest_low)
    slowd = slowk.rolling(window=3).mean()
    return slowk, slowd

def compute_cci(High, Low, Close, window=20):
    tp = (High + Low + Close) / 3
    sma = tp.rolling(window=window).mean()
    mad = tp.rolling(window=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma) / (0.015 * mad)
    return cci

def compute_williams_r(High, Low, Close, window=14):
    highest_high = High.rolling(window=window).max()
    lowest_low = Low.rolling(window=window).min()
    williams_r = -100 * ((highest_high - Close) / (highest_high - lowest_low))
    return williams_r

def compute_mfi(High, Low, Close, Volume, window=14):
    typical_price = (High + Low + Close) / 3
    money_flow = typical_price * Volume
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    positive_flow_sum = positive_flow.rolling(window=window).sum()
    negative_flow_sum = negative_flow.rolling(window=window).sum()
    money_flow_index = 100 - (100 / (1 + (positive_flow_sum / negative_flow_sum)))
    return money_flow_index
