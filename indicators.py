import numpy as np
import pandas as pd
import ta
import pandas_ta as pta
from hmmlearn.hmm import GaussianHMM

import pandas as pd
import ta
import numpy as np

def create_features(df):
    # Hour of the day
    df['hour_of_day'] = df['time'].dt.hour
    
    # Engulfing Candles (Momentum Shift)
    #df['bullish_engulfing'] = (df['close'] > df['open']) & (df['close'].shift(1) < df['open'].shift(1)) & (df['close'] > df['open'].shift(1))
    #df['bearish_engulfing'] = (df['close'] < df['open']) & (df['close'].shift(1) > df['open'].shift(1)) & (df['close'] < df['open'].shift(1))
    
    # Wick Analysis (Rejections)
    #df['upper_wick'] = df['high'] - np.maximum(df['close'], df['open'])
    #df['lower_wick'] = np.minimum(df['close'], df['open']) - df['low']
    #df['candle_body'] = abs(df['close'] - df['open'])
    
    # Volatility & Momentum Indicators
    #df['atr'] = calculate_atr(df, 14)
    #df['rsi'] = calculate_rsi(df, 14)
    df['ema_9'] = calculate_ema(df, 9)
    #df['ema_21'] = calculate_ema(df, 21)
    #df['ema_50'] = calculate_ema(df, 50)
    #df['macd'], df['macd_signal'] = calculate_macd(df)
    #df['vwap'] = calculate_vwap(df)
    #df["price_diff"] = df["close"].diff()

    return df



def calculate_ema(df, period):
    return df['close'].ewm(span=period, adjust=False).mean()

def calculate_macd(df):
    short_ema = df['close'].ewm(span=12, adjust=False).mean()
    long_ema = df['close'].ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_adx(df, period=14):
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()
    plus_dm[plus_dm <= 0] = 0
    minus_dm[minus_dm >= 0] = 0

    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (abs(minus_dm).rolling(window=period).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(window=period).mean()

def calculate_stochastic(df, k_period=14, d_period=3):
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    stoch_k = 100 * ((df['close'] - low_min) / (high_max - low_min))
    stoch_d = stoch_k.rolling(window=d_period).mean()
    return stoch_k, stoch_d

def calculate_roc(df, period=10):
    return ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100

def calculate_cci(df, period=14):
    tp = (df['high'] + df['low'] + df['close']) / 3
    ma = tp.rolling(window=period).mean()
    md = (tp - ma).abs().rolling(window=period).mean()
    return (tp - ma) / (0.015 * md)

def calculate_williams_r(df, period=14):
    highest_high = df['high'].rolling(window=period).max()
    lowest_low = df['low'].rolling(window=period).min()
    return -100 * ((highest_high - df['close']) / (highest_high - lowest_low))

def calculate_momentum(df, period=10):
    return df['close'] - df['close'].shift(period)

def calculate_atr(df, period=14):
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_bollinger_bands(df, period=20, num_std=2):
    sma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    upper_band = sma + (num_std * std)
    lower_band = sma - (num_std * std)
    return upper_band, sma, lower_band

def calculate_obv(df):
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    return obv

def calculate_vwap(df):
    return (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

def calculate_pivot_points(df):
    pivot = (df['high'] + df['low'] + df['close']) / 3
    r1 = (2 * pivot) - df['low']
    s1 = (2 * pivot) - df['high']
    return pivot, r1, s1

# 1. Order Flow / Volume Delta Indicator
def calculate_order_flow(df):
    df['volume_delta'] = df['volume'].diff()
    df['buy_volume'] = np.where(df['close'] > df['open'], df['volume_delta'], 0)
    df['sell_volume'] = np.where(df['close'] < df['open'], df['volume_delta'], 0)
    df['order_flow'] = df['buy_volume'] - df['sell_volume']
    return df

# 2. Chandelier Exit & Parabolic SAR for Trailing Stop
def calculate_chandelier_exit(df, atr_period=22, multiplier=3):
    atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=atr_period)
    high_max = df['high'].rolling(window=atr_period).max()
    low_min = df['low'].rolling(window=atr_period).min()
    df['chandelier_long'] = high_max - (atr * multiplier)
    df['chandelier_short'] = low_min + (atr * multiplier)
    return df

def calculate_parabolic_sar(df):
    psar_df = pta.psar(df['high'], df['low'], acceleration=0.02, maximum=0.2)
    df['parabolic_sar'] = psar_df['PSARl_0.02_0.2']
    return df

def detect_market_regimes(df, n_states=3):
    hmm = GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=1000)

    # Ensure no NaN values before fitting
    returns = np.log(df['close'] / df['close'].shift(1)).dropna()

    # Fit HMM model
    hmm.fit(returns.values.reshape(-1, 1))

    # Predict regimes
    regimes = hmm.predict(returns.values.reshape(-1, 1))

    # Align with original DataFrame length
    df = df.iloc[1:].copy()  # Drop the first row to match length
    df['regime'] = regimes  # Now lengths match

    return df

# Function to compute ERI
def compute_eri(close, high, low, window=14):
    ema = close.ewm(span=window).mean()
    bull_power = high - ema
    bear_power = low - ema
    return bull_power, bear_power

# Function to calculate SAR
def calculate_sar(df, acceleration=0.02, maximum=0.2):
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    sar = np.zeros(len(close))
    af = acceleration
    uptrend = True  # Start with an uptrend (can be dynamically determined)
    ep = high[0]    # Extreme point: highest high for uptrend, lowest low for downtrend
    sar[0] = low[0] # Starting SAR value
    
    for i in range(1, len(close)):
        # Update SAR value
        sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
        
        # Check for trend reversal
        if uptrend:
            if low[i] < sar[i]:  # Trend reversal to downtrend
                uptrend = False
                sar[i] = ep
                af = acceleration
                ep = low[i]
        else:
            if high[i] > sar[i]:  # Trend reversal to uptrend
                uptrend = True
                sar[i] = ep
                af = acceleration
                ep = high[i]
        
        # Update extreme point and acceleration factor
        if uptrend:
            if high[i] > ep:
                ep = high[i]
                af = min(af + acceleration, maximum)
        else:
            if low[i] < ep:
                ep = low[i]
                af = min(af + acceleration, maximum)
    
    return pd.Series(sar, index=df.index)

def compute_fisher_transform(series, window=10):
        min_val = series.rolling(window=window).min()
        max_val = series.rolling(window=window).max()
        value = 2 * ((series - min_val) / (max_val - min_val) - 0.5)
        fisher_transform = 0.5 * np.log((1 + value) / (1 - value))
        return fisher_transform

# Function to calculate support levels
def calculate_support(low, window=14):
    return low.rolling(window=window).min()

# Function to calculate resistance levels
def calculate_resistance(high, low, close, window=14):
    typical_price = (high + low + close) / 3
    return typical_price.rolling(window=window).max()

# Function to classify market conditions based on ADX and Bollinger Bands

def classify_market_conditions(df):

    conditions = []
    for i in range(len(df)):
        if df['ADX'].iloc[i] > 25:
            conditions.append(0)
        elif df['boll_width'].iloc[i] > 0.2:
            conditions.append(1)
        else:
            conditions.append(2)
    df['market_condition'] = conditions
    return df