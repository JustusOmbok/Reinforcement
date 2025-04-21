import gymnasium as gym
import pandas as pd
import MetaTrader5 as mt5
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from gymnasium.wrappers import TimeLimit
import matplotlib.pyplot as plt
from indicators1 import create_features, classify_market_conditions, detect_market_regimes
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
import pickle
from time import time
import numpy as np
import shap
from windowshap import SlidingWindowSHAP
import warnings
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
import torch.nn as nn
warnings.filterwarnings('ignore')
import pandas as pd
from enum import Enum
import matplotlib.pyplot as plt
from ta.trend import EMAIndicator
import sys
import logging
import io
#torch.autograd.set_detect_anomaly(True)
# Reduce matplotlib debug logs
logging.getLogger('PIL').setLevel(logging.WARNING)
# Set up logging for font finding to reduce verbosity
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
# Set matplotlib logging level to WARNING to reduce debug output
logging.getLogger('matplotlib').setLevel(logging.WARNING)
# Ensure UTF-8 encoding for console output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

logging.basicConfig(
    level=logging.WARNING,  # Change to logging.INFO to hide debug logs
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training.log", encoding='utf-8'),  # Save logs to a file
        logging.StreamHandler(sys.stdout)  # Explicitly print logs to console
    ]
)

logger = logging.getLogger(__name__)  # Get the logger



class Actions(Enum):
    Hold = 2
    Sell = 0
    Buy = 1


class Positions(Enum):
    Short = 0
    Long = 1

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long


class TradingEnv(gym.Env):
    def __init__(self, df, window_size, frame_bound, unit_side='left', render_mode=None):
        assert df.ndim == 2
        assert render_mode is None or render_mode in self.metadata['render_modes']
        assert len(frame_bound) == 2
        assert unit_side.lower() in ['left', 'right']

        self.frame_bound = frame_bound
        self.unit_side = unit_side.lower()
        super().__init__()

        self.trade_fee = 0.0003  # unit
        self.render_mode = render_mode

        if 'hour_of_day' not in df.columns:  
            if 'time' in df.columns:
                df['hour_of_day'] = df['time'].dt.hour.astype(pd.Int64Dtype())  
            else:
                raise ValueError("Missing 'time' column in dataframe! Cannot extract 'hour_of_day'.")
        else:
            logger.info("Using existing 'hour_of_day' column.")

        if df['hour_of_day'].isnull().any():
            logger.warning("Warning: NaN values found in 'hour_of_day'. Replacing with 0.")
            df['hour_of_day'].fillna(0, inplace=True)

        df['hour_of_day'] = df['hour_of_day'].astype(int)
        self.df = df
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        # Spaces
        self.action_space = gym.spaces.Discrete(len(Actions))
        INF = 1e10
        self.observation_space = gym.spaces.Box(
            low=-INF, high=INF, shape=self.shape, dtype=np.float32,
        )

        # ✅ Initialize episode tracking
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._truncated = False
        self._current_tick = self._start_tick  # ✅ Ensure it's initialized
        self._last_trade_tick = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None

        self.entry_price = None
        self.active_trades = []  
        self.stop_loss = None
        self.take_profit = None
        self._trade_active = False
        self.hourly_losses = {hour: 0 for hour in range(24)}
        self.hourly_profits = {hour: 0 for hour in range(24)}
        self.entry_hour = None  

        self.trade_rewards = {}  # ✅ FIXED: Initialize trade rewards dictionary
        self.trade_actions = {}
        self.trailing_stop = None
        self.ema_5 = EMAIndicator(close=df['close'], window=5).ema_indicator()
        self.ema_10 = EMAIndicator(close=df['close'], window=10).ema_indicator()
        self.ema_25 = EMAIndicator(close=df['close'], window=25).ema_indicator()
        self.ema_50 = EMAIndicator(close=df['close'], window=50).ema_indicator()
        self.parabolic_sar = self.calculate_sar(self.df)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.action_space.seed(int((self.np_random.uniform(0, seed if seed is not None else 1))))

        self._truncated = False
        self._current_tick = self._start_tick  # ✅ Ensure we start at the correct tick
        self._last_trade_tick = max(self._current_tick - 1, 0)
        self._position = Positions.Short
        self._position_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.
        self._total_profit = 0.  
        self._first_rendering = True
        self.history = {
            'total_reward': [],
            'total_profit': [],
            'position': [],
            'reward': [],
        }
        self._trade_active = False  
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.trailing_stop = None
        self.trade_rewards = {}  # ✅ Reset trade rewards when resetting env
        self.trade_actions = {}
        self.active_trades = []  # ✅ Reset active trades

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info


    def step(self, action):
        """
        Executes trades, checks for exits, and assigns rewards to their opening tick.
        Fix: Rewards are now correctly mapped to their specific entry tick.
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}. Action must be within the bounds of the action space: {self.action_space}.")
        self._current_tick += 1

        if self._current_tick >= len(self.prices):
            self._truncated = True
            return self._get_observation(), {}, self._truncated, self._truncated, self._get_info()

        self._truncated = self._end_tick == self._current_tick

        if self._should_trade(action):
            self._execute_trade(action)  
            self.trade_actions[self._current_tick] = action  # ✅ Store action with its entry tick

        self._check_trade_exit()  # ✅ Process trade exits first

        latest_obs = self._get_observation()  # ✅ Always get the latest observation for current tick
        logger.debug(f"[DEBUG] Tick {self._current_tick}: Latest Observation Retrieved.")

        if not self.trade_rewards:
            logger.debug(f"[DEBUG] Tick {self._current_tick}: No trade exited. Returning latest observation.")
            return latest_obs, {}, False, False, self._get_info()  # ✅ Return empty dictionary if no trade exited

        # ✅ Process rewards while maintaining correct entry tick
        rewards_info = {tick: round(self.trade_rewards.pop(tick), 2) for tick in list(self.trade_rewards.keys())}
        logger.debug(f"[DEBUG] Step rewards: {rewards_info}")

        return latest_obs, rewards_info, self._truncated, self._truncated, self._get_info()

    
    def _calculate_rsi(self, window=14):
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[self._current_tick]
    
    def calculate_sar(self, df, acceleration=0.02, maximum=0.2):
        high = df['high'].values  # Get full arrays
        low = df['low'].values
        close = df['close'].values
        sar = np.zeros(len(close))  # SAR array initialization
        
        # Determine initial trend
        uptrend = high[0] > low[0]  # Assume uptrend if first high > first low
        ep = high[0] if uptrend else low[0]  # Set initial Extreme Point
        sar[0] = low[0] if uptrend else high[0]  # Set initial SAR
        af = acceleration  # Acceleration factor

        for i in range(1, len(close)):
            # Calculate new SAR
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])

            # Ensure SAR does not cross price (enforce valid bounds)
            if uptrend:
                sar[i] = min(sar[i], min(low[i - 1], low[i]))  # SAR must be below price in an uptrend
            else:
                sar[i] = max(sar[i], max(high[i - 1], high[i]))  # SAR must be above price in a downtrend

            # Update EP and detect reversals
            if uptrend:
                if high[i] > ep:
                    ep = high[i]  # New highest high
                    af = min(af + acceleration, maximum)  # Increase AF
                if low[i] < sar[i]:  # Downtrend reversal
                    uptrend = False
                    sar[i] = ep  # Set SAR to previous EP
                    ep = low[i]  # Reset EP to lowest low
                    af = acceleration  # Reset AF
            else:
                if low[i] < ep:
                    ep = low[i]  # New lowest low
                    af = min(af + acceleration, maximum)  # Increase AF
                if high[i] > sar[i]:  # Uptrend reversal
                    uptrend = True
                    sar[i] = ep  # Set SAR to previous EP
                    ep = high[i]  # Reset EP to highest high
                    af = acceleration  # Reset AF

        return pd.Series(sar, index=df.index)
    
    def _calculate_adx(self, period=14):
        """Calculates the Average Directional Index (ADX) to measure trend strength."""
        df = self.df.copy()

        # ✅ Calculate True Range (TR)
        df['high-low'] = df['high'] - df['low']
        df['high-close'] = abs(df['high'] - df['close'].shift(1))
        df['low-close'] = abs(df['low'] - df['close'].shift(1))
        df['TR'] = df[['high-low', 'high-close', 'low-close']].max(axis=1)

        # ✅ Calculate Directional Movement (DM+ and DM-)
        df['DM+'] = df['high'].diff()
        df['DM-'] = df['low'].diff()
        df['DM+'] = np.where((df['DM+'] > df['DM-']) & (df['DM+'] > 0), df['DM+'], 0)
        df['DM-'] = np.where((df['DM-'] > df['DM+']) & (df['DM-'] > 0), -df['DM-'], 0)

        # ✅ Smooth the values using Exponential Moving Average (EMA)
        df['TR_smooth'] = df['TR'].ewm(span=period, adjust=False).mean()
        df['DM+_smooth'] = df['DM+'].ewm(span=period, adjust=False).mean()
        df['DM-_smooth'] = df['DM-'].ewm(span=period, adjust=False).mean()

        # ✅ Calculate the Directional Indicators (DI+ and DI-)
        df['DI+'] = np.where(df['TR_smooth'] == 0, 0, (df['DM+_smooth'] / df['TR_smooth']) * 100)
        df['DI-'] = np.where(df['TR_smooth'] == 0, 0, (df['DM-_smooth'] / df['TR_smooth']) * 100)

        # ✅ Calculate the Directional Index (DX)
        df['DX'] = np.where((df['DI+'] + df['DI-']) == 0, 0, abs(df['DI+'] - df['DI-']) / (df['DI+'] + df['DI-']) * 100)

        # ✅ Calculate ADX as the smoothed DX over the period
        df['ADX'] = abs(df['DX'].ewm(span=period, adjust=False).mean())  # Always positive

        # ✅ Return the most recent ADX value
        return df['ADX'].iloc[self._current_tick - 1]

    
    def wait_for_next_execution(self):
        current_df_index = self._current_tick - self.window_size + self.frame_bound[0]
        hour_of_day = self.df['hour_of_day'].iloc[current_df_index]
    
        if (6 <= hour_of_day <= 19):
            return True
        else:
            return False

    
    def _should_trade(self, action):
        ema_5 = self.ema_5.iloc[self._current_tick]
        ema_10 = self.ema_10.iloc[self._current_tick]
        
        if action == Actions.Buy.value:
            return True
        elif action == Actions.Sell.value:
            return True
        return False


    def _execute_trade(self, action):
        """Executes a trade and stores its opening tick for reward assignment."""
        if action == Actions.Buy.value:
            position = Positions.Long
        elif action == Actions.Sell.value:
            position = Positions.Short
        else:
            return

        # Set entry price, stop loss, take profit
        current_price = self.prices[self._current_tick]
        current_df_index = self._current_tick - self.window_size + self.frame_bound[0]
        moving_average = self.df['close'].iloc[current_df_index - self.window_size:current_df_index].mean()
        lookback_period = self.window_size
        spread = 0.0003  # Spread for EURUSD
        entry_price = current_price + spread if position == Positions.Long else current_price - spread
        atr = self._calculate_atr()

        stop_loss = entry_price - (atr) if position == Positions.Long else entry_price + (atr)
        take_profit = entry_price + (atr) if position == Positions.Long else entry_price - (atr)

        # Store trade with entry tick
        trade = {
            "position": position,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "open_tick": self._current_tick,
            "exit_price": None,  # Track exit price when trade is closed
            "active": True  # Keep trade active until exit conditions are met
        }

        self.active_trades.append(trade)

        logger.info(f"Trade opened | Tick: {self._current_tick} | Entry: {entry_price} | SL: {stop_loss} | TP: {take_profit} | Position: {position}")
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        # ✅ Store entry hour for hourly tracking
        self.entry_hour = self.df['hour_of_day'].iloc[current_df_index]


    def _check_trade_exit(self):
        """Checks if trades should exit and stores rewards correctly."""
        if not self.active_trades:
            return []
        
        closed_trades = []

        for trade in self.active_trades:
            if not trade["active"]:
                continue  # Skip already closed trades

            current_price = self.prices[self._current_tick]
            current_high = self.df['high'].iloc[self._current_tick]
            current_low = self.df['low'].iloc[self._current_tick]
            exit_price = None
            reward = 0

            # Check SL/TP every tick **until it's hit**
            if trade["position"] == Positions.Long:
                if current_low <= trade["stop_loss"]:
                    exit_price = trade["stop_loss"]
                elif current_high >= trade["take_profit"]:
                    exit_price = trade["take_profit"]
            elif trade["position"] == Positions.Short:
                if current_high >= trade["stop_loss"]:
                    exit_price = trade["stop_loss"]
                elif current_low <= trade["take_profit"]:
                    exit_price = trade["take_profit"]

            # Exit trade only when SL/TP is hit
            if exit_price is not None:
                trade["exit_price"] = exit_price  # Track exit price
                trade["active"] = False  # Mark trade as inactive
                reward = (exit_price - trade["entry_price"]) * 10000 if trade["position"] == Positions.Long else (trade["entry_price"] - exit_price) * 10000  # Pips


                self.trade_rewards[trade["open_tick"]] = reward
                
                logger.debug(f"[DEBUG] Trade exited | Open Tick: {trade['open_tick']} | Exit Tick: {self._current_tick} | Exit Price: {exit_price} | Current Price: {current_price} | Reward: {reward:.2f}")
                logger.debug(f"[DEBUG] Stored reward {reward:.2f} at tick {trade['open_tick']}")
                
                closed_trades.append(trade)
                
                logger.debug(f"Trade exited | Open Tick: {trade['open_tick']} | Exit Tick: {self._current_tick} | Profit: {reward:.2f} | Position: {trade['position']}")

        # Remove closed trades from active list
        self.active_trades = [t for t in self.active_trades if t["active"]]  # Keep only active trades


    def _calculate_atr(self):
        # Simplified calculation of Average True Range (ATR)
        high = self.df['high'][self._current_tick - self.window_size:self._current_tick].max()
        low = self.df['low'][self._current_tick - self.window_size:self._current_tick].min()
        close_prev = self.df['close'].iloc[max(self._current_tick - 1, 0)]
        return max(high - low, abs(high - close_prev), abs(low - close_prev))
    
    def _get_info(self):
        return dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            position=self._position
        )
    
    def _get_observation_at_tick(self, tick):
        """Retrieves the observation corresponding to a specific past tick (entry tick)."""
        return self.signal_features[tick - self.window_size:tick]


    def _get_observation(self):
        return {
            "signal_features": self.signal_features[self._current_tick - self.window_size:self._current_tick],
            "entry_tick": self._current_tick
        }

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def _process_data(self):
        prices = self.df.loc[:, 'close'].to_numpy()

        prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = self.df.loc[:, self.df.columns.difference(['time', 'close', 'low', 'high', 'open', 'volume', 'spread'])].replace([np.inf, -np.inf], np.nan).ffill().bfill().to_numpy()  # drop time and price columns and ensure no NaN values are present by filling foward and backward
        #print(pd.DataFrame(signal_features).head(20))

        return prices, signal_features





# Function to fetch historical data from MetaTrader5
def fetch_data(symbol, timeframe, start, end):
    rates = mt5.copy_rates_range(symbol, timeframe, start, end)

    if rates is None or len(rates) == 0:
        logger.error(f"No data found for symbol {symbol} in the given date range.")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], unit='s')

    # fetch data between (6 <= hour_of_day <= 19):
    df['hour_of_day'] = df['time'].dt.hour.astype(pd.Int64Dtype())  # Ensure hour_of_day is in the DataFrame
    df = df[(df['hour_of_day'] >= 6) & (df['hour_of_day'] <= 19)]  # Filter data between 6 AM and 7 PM


    # Ensure volume is in the DataFrame
    
    df['volume'] = df['tick_volume']

    df = df.drop(['real_volume', 'tick_volume'], axis=1)

    return df

def calculate_sharpe_ratio(returns):
    if len(returns) == 0:
        return 0  # Avoid division by zero
    std_returns = np.std(returns)
    if std_returns < 1e-10:
        return np.sign(np.mean(returns)) * np.inf  # If returns are constant, Sharpe should be high or low
    return np.mean(returns) / std_returns * np.sqrt(252)

def calculate_max_drawdown(balance_history):
    peak = balance_history[0]
    max_drawdown = 0

    for balance in balance_history:
        peak = max(peak, balance)  # Track the highest balance reached
        drawdown = balance - peak  # Compute drawdown from peak
        max_drawdown = min(max_drawdown, drawdown)  # Store the worst drawdown

    return max_drawdown

def test_rl_agent(model, env, num_episodes=1):
    balance_history = [0.0]  # Start balance at 0
    profits, losses = [], []
    hourly_losses = {hour: 0 for hour in range(24)}
    hourly_profits = {hour: 0 for hour in range(24)}

    for _ in range(num_episodes):
        reset_result = env.reset()

        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result  
        done = False

        while not done:
            action, _ = model.predict(obs)
            obs, reward_info, done, _, _ = env.step(action)

            for entry_tick, reward_value in reward_info.items():
                if reward_value == 0:
                    continue

                trade_hour = env.df['hour_of_day'].iloc[entry_tick - env.window_size] if 'hour_of_day' in env.df.columns else 0
                logger.debug(f"[DEBUG] Received Reward {reward_value:.2f} at Tick {entry_tick} (Hour {trade_hour})")

                reward = round(reward_value, 2)
                new_balance = balance_history[-1] + reward
                balance_history.append(new_balance)
                logger.info(f"Step Reward = {reward:.2f}, Balance = {new_balance:.2f}")

                if reward > 0:
                    profits.append(reward)
                    hourly_profits[trade_hour] += reward
                elif reward < 0:
                    losses.append(reward)
                    hourly_losses[trade_hour] += abs(reward)

    net_profit = balance_history[-1]
    total_trades = len(profits) + len(losses)
    win_rate = len(profits) / total_trades if total_trades > 0 else 0
    average_trade = net_profit / total_trades if total_trades > 0 else 0
    total_loss = abs(sum(losses))
    profit_factor = sum(profits) / total_loss if total_loss > 1e-10 else np.nan
    max_drawdown = calculate_max_drawdown(balance_history)
    returns = np.diff(balance_history) / np.maximum(np.abs(balance_history[:-1]), 1e-10)
    sharpe_ratio = calculate_sharpe_ratio(returns) if len(returns) > 1 else np.nan

    logger.warning("\n=== Hourly Losses/Profits Recorded ===")
    for hour in range(24):
        if hourly_losses[hour] > 0 or hourly_profits[hour] > 0:
            logger.warning(f"Hour {hour}: Loss = {hourly_losses[hour]:.2f}, Profit = {hourly_profits[hour]:.2f}")


    logger.warning("\n=== Test Results ===")
    logger.warning(f"Net Profit: {net_profit:.2f}")
    logger.warning(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    logger.warning(f"Profit Factor: {profit_factor:.2f}")
    logger.warning(f"Win Rate: {win_rate:.2%}")
    logger.warning(f"Profit Trades: {len(profits)}")
    logger.warning(f"Loss Trades: {len(losses)}")
    logger.warning(f"Total Trades: {total_trades}")
    logger.warning(f"Average Trade: {average_trade:.2f}")
    logger.warning(f"Max Drawdown: {max_drawdown:.2f}")

    return {
        "Net Profit": net_profit,
        "Sharpe Ratio": sharpe_ratio,
        "Profit Factor": profit_factor,
        "Win Rate": win_rate,
        "Profit Trades": len(profits),
        "Loss Trades": len(losses),
        "Total Trades": total_trades,
        "Average Trade": average_trade,
        "Max Drawdown": max_drawdown,
        "Balance History": balance_history,
    }



def plot_balance_curve(balance_history):
    plt.plot(balance_history)
    plt.title('P&L Curve')
    plt.xlabel('Steps')
    plt.ylabel('Balance')
    plt.show()

# Calculate Sliding Window SHAP values
def calculate_shap_values(model, env, num_samples='auto'):
    env.reset()
    
    B_ts = env.signal_features[:env.window_size * (env.signal_features.shape[0] // env.window_size)].reshape(-1, env.window_size, env.signal_features.shape[1])
    summarized_B_ts = shap.sample(B_ts, 100)  # Summarize to 1000 samples
    shap_values = SlidingWindowSHAP(model, stride=5, window_len=5, B_ts=summarized_B_ts, test_ts=B_ts).shap_values(num_output=2, nsamples=num_samples)

    # Print SHAP values
    feature_names = env.df.columns.difference(['time', 'close', 'low', 'high', 'open', 'volume', 'spread'])
    shap_values_mean = shap_values[0].mean(axis=0)  # Average SHAP values for each feature
    logger.info("Feature Importance (SHAP Values):")
    for feature, value in zip(feature_names, shap_values_mean):
        logger.info(f"{feature}: {value:.4f}")

    # Plot SHAP values importance for the output 
    plt.figure(figsize=(10, 5))
    plt.barh(range(len(shap_values[0].mean(axis=0))), shap_values[0].mean(axis=0))
    plt.yticks(range(len(shap_values[0].mean(axis=0))), feature_names)
    plt.title('SHAP Values Importance')
    plt.show()

    return shap_values


class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_size=128):
        super(ActorCritic, self).__init__()

        # Recurrent Layer (Replaces LSTM with GRU)
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)

        # Fully Connected Layers
        self.actor = nn.Sequential(
            nn.LayerNorm(hidden_size),  # Normalization to stabilize training
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)  # Ensures valid probability distribution
        )

        self.critic = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1, x.size(-1))  # Reshape to (batch_size, seq_len, input_dim)
        x, _ = self.gru(x.clone())  # ✅ Prevents in-place modification
        x = x[:, -1, :]  # Take the last output

        policy = self.actor(x)  # Action probabilities
        value = self.critic(x)  # State value estimation
        return Categorical(policy), value

    def predict(self, obs):
        if isinstance(obs, dict) and "signal_features" in obs:
            features = obs["signal_features"]
            obs = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        else:
            print("Error: obs is not a dictionary or missing 'signal_features'")
            raise ValueError("Invalid obs format")
        with torch.no_grad():
            dist, _ = self.forward(obs)
        action = dist.sample().item()
        return action, np.array([[action]])  # Tuple: first for trading, second for SHAP


class MultiAgentA3C:
    def __init__(self, env, num_agents=3, gamma=0.99, lr=1e-4):
        self.env = env
        self.test_env = test_env  # Store test environment
        self.num_agents = num_agents
        self.gamma = gamma

        # Initialize models and optimizers
        self.models = [ActorCritic(env.observation_space.shape[1], env.action_space.n) for _ in range(num_agents)]
        self.actor_optimizers = [optim.Adam(model.actor.parameters(), lr=lr) for model in self.models]
        self.critic_optimizers = [optim.Adam(model.critic.parameters(), lr=lr) for model in self.models]

    def predict(self, obs):
        if isinstance(obs, tuple):  
            obs = obs[0]
        obs = torch.tensor(obs["signal_features"], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist, _ = self.models[0](obs)  # Use first model for inference
        return dist.sample().item(), None  

    def train(self, num_episodes=100):
        """
        Train the multi-agent A3C model.
        Fix: Rewards are now stored and retrieved using the correct entry tick.
        """
        best_winrate = -np.inf
        patience_counter = 10

        for episode in range(num_episodes):
            logger.info(f"Episode {episode}")
            total_rewards = []

            for agent_id, model in enumerate(self.models):
                reset_result = self.env.reset()
                obs, _ = reset_result if isinstance(reset_result, tuple) else (reset_result, None)

                log_probs = {}  # Store log probs per entry tick
                values = {}  # Store value estimates per entry tick
                rewards = {}  # Store rewards per entry tick

                done = False  # Ensure done is defined before the loop
                while not done:
                    signal_features = torch.tensor(obs["signal_features"], dtype=torch.float32).unsqueeze(0)
                    entry_tick = obs["entry_tick"]
                    
                    dist, value = model(signal_features)  # Get action distribution & value estimate
                    action = dist.sample()  # Sample action
                    log_prob = dist.log_prob(action)  # Log probability of the chosen action

                    # Store log_prob & value directly using entry_tick
                    log_probs[entry_tick] = log_prob
                    values[entry_tick] = value

                    obs, reward_info, done, _, _ = self.env.step(action.item())  # Take action

                    if not isinstance(reward_info, dict):
                        reward_info = {}

                    # Now we know the correct entry_tick from reward_info, store corresponding values
                    for tick, reward in reward_info.items():
                        if tick not in log_probs:  
                            log_probs[tick] = log_probs[entry_tick]
                            values[tick] = values[entry_tick]

                        rewards[tick] = reward  # Store the reward

                        logger.debug(f"Training Method = Entry Tick: {tick}, Action: {action.item()}, Reward: {reward}")

                self.optimize(log_probs, values, rewards, agent_id)
                total_rewards.append(sum(rewards.values()))

            metrics = test_rl_agent(self, self.test_env, num_episodes=5)

            # Track best model
            if metrics['Win Rate'] > best_winrate:
                best_winrate = metrics['Win Rate']
                patience_counter = 0
                logger.warning("Improved Balance History. Saving model...")
                torch.save(self.models[0], "best_model.pth")
                with open("scaler.pkl", "wb") as f:
                    pickle.dump(scaler, f)
            else:
                patience_counter += 1
                logger.warning(f"No improvement. Patience counter: {patience_counter}")

            if patience_counter >= 10:  # Early stopping
                logger.warning("Early stopping triggered.")
                break

    def optimize(self, log_probs, values, rewards, agent_id):
        returns = {}
        discounted_sum = 0

        # Compute discounted returns
        for entry_tick in sorted(rewards.keys(), reverse=True):  
            discounted_sum = rewards[entry_tick] + self.gamma * discounted_sum
            returns[entry_tick] = discounted_sum

        # Optimization loop
        for entry_tick in rewards.keys():
            log_prob = log_probs[entry_tick]
            value = values[entry_tick]

            # ✅ Add Debugging
            if not value.requires_grad:
                logger.warning(f"🚨 WARNING: value at tick {entry_tick} does NOT require gradients!")

            # Ensure return_val does NOT require gradients
            return_val = torch.tensor([returns[entry_tick]], dtype=torch.float32, device=value.device, requires_grad=False)

            # Compute advantage safely
            advantage = (return_val - value).clone()

            # Compute policy loss
            policy_loss = (-log_prob * advantage.detach()).mean()

            # Compute value loss
            value_loss = nn.functional.mse_loss(value, return_val.detach())

            # 🚀 Debugging Before `.backward()`
            #logger.info(f"Optimizing Agent {agent_id} | Entry Tick {entry_tick}")
            #logger.info(f"  - Policy Loss: {policy_loss.item():.5f}, Value Loss: {value_loss.item():.5f}")

            # ✅ Check Tensor Versions Before `.backward()`
            #logger.info(f"  🔍 Checking tensor versions BEFORE backpropagation...")
            #logger.info(f"  - value tensor version: {value._version if hasattr(value, '_version') else 'N/A'}")
            #logger.info(f"  - return_val tensor version: {return_val._version if hasattr(return_val, '_version') else 'N/A'}")

            # ✅ Backpropagate Safely
            total_loss = policy_loss + value_loss
            self.actor_optimizers[agent_id].zero_grad()
            self.critic_optimizers[agent_id].zero_grad()

            total_loss.backward(retain_graph=True)  # 🔥 If this fails, we know exactly what changed

            # ✅ Debugging Gradient Norms
            #for name, param in self.models[agent_id].named_parameters():
                #if param.grad is not None:
                    #logger.info(f"  ✅ Gradient for {name}: {param.grad.norm().item():.5f}")


if __name__ == "__main__":
    # Initialize MetaTrader 5 connection and fetch data
    mt5.initialize(login=100003703, server="FBS-Demo", password="jCv0+4N.")
    df = fetch_data('EURUSD', mt5.TIMEFRAME_H1, start=datetime.datetime(2019, 9, 1), end=(datetime.datetime.now() + datetime.timedelta(hours=4)))
    

    # Preprocess data
    logger.info(df.tail())
    df = create_features(df)
    #df = classify_market_conditions(df)
    #df = detect_market_regimes(df)
    logger.info(df.shape)
    logger.warning(df.tail())
    #print(df.isna().sum())
    scaler = StandardScaler()
    feature_columns = df.columns.difference(['time', 'close', 'low', 'high', 'open', 'volume', 'spread', 'hour_of_day', 'regime', 'market_condition', 'bullish_signal', 'bearish_signal'])
    
    df[feature_columns] = df[feature_columns].replace([np.inf, -np.inf], np.nan)
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    # Save feature names for importance calculation

    tscv = TimeSeriesSplit(n_splits=5)

    for train_idx, test_idx in tscv.split(df):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]


    # Initialize environments
    
    train_env = TradingEnv(train_df, window_size=14, frame_bound=(14, len(train_df)), unit_side='left')
    test_env = TradingEnv(test_df, window_size=14, frame_bound=(14, len(test_df)), unit_side='left')

    # Train RL agent
    model = MultiAgentA3C(train_env)
    model.train(num_episodes=100)

    model = torch.load("best_model.pth")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Test RL agent
    #test_rl_agent(model, env=test_env, num_episodes=10)

    # Plot balance curve
    test_results = test_rl_agent(model, env=test_env, num_episodes=5)
    plot_balance_curve(test_results['Balance History'])

    # Calculate SHAP values
    #shap_values = calculate_shap_values(model, test_env, num_samples=1000)

    # Close MetaTrader 5 connection
    mt5.shutdown()
