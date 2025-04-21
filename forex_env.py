import pandas as pd
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import gym
from gym import spaces
from ta.trend import EMAIndicator
import sys
import logging
import io


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
            print("Using existing 'hour_of_day' column.")

        if df['hour_of_day'].isnull().any():
            print("Warning: NaN values found in 'hour_of_day'. Replacing with 0.")
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
        print(f"[DEBUG] Tick {self._current_tick}: Latest Observation Retrieved.")

        if not self.trade_rewards:
            print(f"[DEBUG] Tick {self._current_tick}: No trade exited. Returning latest observation.")
            return latest_obs, {}, False, False, self._get_info()  # ✅ Return empty dictionary if no trade exited

        # ✅ Process rewards while maintaining correct entry tick
        rewards_info = {tick: round(self.trade_rewards.pop(tick), 2) for tick in list(self.trade_rewards.keys())}
        print(f"[DEBUG] Step rewards: {rewards_info}")

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
        self.df['hour_of_day'] = self.df['time'].dt.hour
        hour_of_day = self.df['hour_of_day'].iloc[current_df_index]
    
        if (6 <= hour_of_day <= 19):
            return True
        else:
            return False

    
    def _should_trade(self, action):
        ema_5 = self.ema_5.iloc[self._current_tick]
        ema_10 = self.ema_10.iloc[self._current_tick]
        
        if action == Actions.Buy.value and ema_5 > ema_10:
            return True
        elif action == Actions.Sell.value and ema_5 < ema_10:
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

        stop_loss = entry_price - (atr * 1.5) if position == Positions.Long else entry_price + (1.5)
        take_profit = entry_price + (2 * atr) if position == Positions.Long else entry_price - (2 * atr)

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

        print(f"Trade opened | Tick: {self._current_tick} | Entry: {entry_price} | SL: {stop_loss} | TP: {take_profit} | Position: {position}")
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
                
                print(f"[DEBUG] Trade exited | Open Tick: {trade['open_tick']} | Exit Tick: {self._current_tick} | Exit Price: {exit_price} | Current Price: {current_price} | Reward: {reward:.2f}")
                print(f"[DEBUG] Stored reward {reward:.2f} at tick {trade['open_tick']}")
                
                closed_trades.append(trade)
                
                print(f"Trade exited | Open Tick: {trade['open_tick']} | Exit Tick: {self._current_tick} | Profit: {reward:.2f} | Position: {trade['position']}")

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