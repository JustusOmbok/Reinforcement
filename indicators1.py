import numpy as np
import pandas as pd
import ta
import pandas_ta as pta
from hmmlearn.hmm import GaussianHMM

# Main feature creation function with all updates
def create_features(df):
    df = df.copy()  # Avoid modifying the original DataFrame
    df['time'] = pd.to_datetime(df['time'])

    df['ema_9'] = calculate_ema(df, 9)
    df['ema_21'] = calculate_ema(df, 21)
    df['macd'], df['macd_signal'] = calculate_macd(df)
    df['rsi'] = calculate_rsi(df, 14)
    df['adx'] = calculate_adx(df, 14)
    df['stoch_k'], df['stoch_d'] = calculate_stochastic(df, 14, 3)
    df['roc'] = calculate_roc(df, 10)
    df['cci'] = calculate_cci(df, 14)
    df['williams_r'] = calculate_williams_r(df, 14)
    df['momentum'] = calculate_momentum(df, 10)
    df['atr'] = calculate_atr(df, 14)
    df['upper_band'], df['middle_band'], df['lower_band'] = calculate_bollinger_bands(df, 20, 2)
    df['obv'] = calculate_obv(df)
    df['vwap'] = calculate_vwap(df)
    df['pivot'], df['r1'], df['s1'] = calculate_pivot_points(df)

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

def detect_market_regimes(df, n_states=3):
    hmm = GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=1000)

    # **Ensure no NaNs before training**
    df = df.dropna().reset_index(drop=True)

    # **Ensure returns are correctly calculated**
    returns = np.log(df['close'].iloc[1:] / df['close'].iloc[:-1].values).dropna()

    # **Train HMM model**
    hmm.fit(returns.values.reshape(-1, 1))

    # **Predict regimes**
    regimes = hmm.predict(returns.values.reshape(-1, 1))

    # **Align output with DataFrame length**
    df = df.iloc[1:].copy()  # Drop first row to match `returns`
    df['regime'] = regimes  # Now correctly aligned

    return df


def classify_market_conditions(df):
    conditions = np.zeros(len(df))  # Pre-allocate conditions array

    for i in range(len(df)):
        boll_width = (df['upper_band'].iloc[i] - df['lower_band'].iloc[i]) / df['middle_band'].iloc[i] if df['middle_band'].iloc[i] != 0 else np.nan
        df['boll_width'].iloc[i] = boll_width  # Update the DataFrame with the calculated value
        # **Check for NaN values before accessing indicators
        if df['adx'].iloc[i] > 25:
            conditions[i] = 0  # Trending Market
        elif df['boll_width'].iloc[i] > 0.2:
            conditions[i] = 1  # Volatile Market
        else:
            conditions[i] = 2  # Range-bound Market

    df['market_condition'] = conditions.astype(int)  # Convert to integers
    return df
