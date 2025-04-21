from time import time
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import gymnasium as gym
import numpy as np


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

    metadata = {'render_modes': ['human'], 'render_fps': 3}

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

        self.df = df
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        # spaces
        self.action_space = gym.spaces.Discrete(len(Actions))
        INF = 1e10
        self.observation_space = gym.spaces.Box(
            low=-INF, high=INF, shape=self.shape, dtype=np.float32,
        )

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._truncated = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None
        #self.cool_down = 3  # Cool-down period in candles
        #self._cool_down_counter = 0  # Counter for cool-down
        self.entry_price = None
        self.stop_loss = None
        self.take_profit1 = None
        self.take_profit2 = None
        self._trade_active = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.action_space.seed(int((self.np_random.uniform(0, seed if seed is not None else 1))))

        self._truncated = False
        self._current_tick = self._start_tick
        self._last_trade_tick = max(self._current_tick - 1, 0)
        self._position = Positions.Short
        self._position_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.
        self._total_profit = 0.  # unit
        self._first_rendering = True
        self.history = {
            'total_reward': [],
            'total_profit': [],
            'position': [],
            'reward': [],
        }
        self._trade_active = False  # Clear any residual trades
        self.entry_price = None
        self.stop_loss = None
        self.take_profit1 = None
        self.take_profit2 = None

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info

    def step(self, action):
        assert self.action_space.contains(action)
        self._current_tick += 1
        self._truncated = self._current_tick == self._end_tick

        reward = 0  # Initialize reward to 0

        # Skip new trade actions if a trade is already active
        if self._trade_active:
            reward = self._check_trade_exit()
        elif self._should_trade(action):
            self._execute_trade(action)

        self._update_history(self._get_info())
        observation = self._get_observation()

        if self.render_mode == 'human':
            self._render_frame()

        info = self._get_info()
        terminated = self._truncated
        return observation, reward, terminated, self._truncated, info
    
    def _calculate_rsi(self, window=14):
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[self._current_tick]

    def _is_favorable_condition(self):
        """Improved trade entry conditions with volume and trend confirmation."""
        current_price = self.prices[self._current_tick]
        moving_average = self.df['close'][self._current_tick - self.window_size:self._current_tick].mean()
        rsi = self._calculate_rsi()
        stddev = self.df['close'][self._current_tick - self.window_size:self._current_tick].std()
        lower_band = moving_average - 2 * stddev
        upper_band = moving_average + 2 * stddev
        atr = self._calculate_atr()
        current_df_index = self._current_tick - self.window_size + self.frame_bound[0]
        volume = self.df['volume'].iloc[current_df_index]
        avg_volume = self.df['volume'].iloc[self._current_tick - self.window_size:self._current_tick].mean()
        ema_short = self.df['ema_short'].iloc[current_df_index]
        ema_long = self.df['ema_long'].iloc[current_df_index]

        #if atr > 4 or volume < avg_volume:
            #return False  # Avoid high volatility & low volume trades

        if self._position == Positions.Short and rsi < 30 and ema_short > ema_long:
            return True
        elif self._position == Positions.Long and rsi > 70 and ema_short < ema_long:
            return True
        elif self._position == Positions.Short and current_price < lower_band:
            return True
        elif self._position == Positions.Long and current_price > upper_band:
            return True
        elif self._position == Positions.Short and current_price < moving_average:
            return True
        elif self._position == Positions.Long and current_price > moving_average:
            return True
        
        return False



    def wait_for_next_execution(self):
        current_df_index = self._current_tick - self.window_size + self.frame_bound[0]
        self.df['hour_of_day'] = self.df['time'].dt.hour
        hour_of_day = self.df['hour_of_day'].iloc[current_df_index]
    
        if (3 <= hour_of_day <= 21): # if hour of day is between 3 and 21
            return True
        else:
            return False
            


    def _should_trade(self, action):
        return (
            (action == Actions.Buy.value and self._position == Positions.Short and self.wait_for_next_execution()) or
            (action == Actions.Sell.value and self._position == Positions.Long and self.wait_for_next_execution())
        )


    def _execute_trade(self, action):
        self._position = Positions.Long if action == Actions.Buy.value else Positions.Short
        self._enter_trade()
        self._last_trade_tick = self._current_tick
        self._trade_active = True

    
    def _enter_trade(self):
        current_price = self.prices[self._current_tick]
        if self._position == Positions.Long:
            self.entry_price = current_price + 0.35
        else:
            self.entry_price = current_price - 0.35

        atr = self._calculate_atr()
        #if atr >= 4:
            #atr = 4

        stop_loss_multiplier = 1
        self.stop_loss = (
            self.entry_price - (stop_loss_multiplier * atr)
            if self._position == Positions.Long
            else self.entry_price + (stop_loss_multiplier * atr)
        )

        self.trailing_stop = self.stop_loss  # Initialize trailing stop
        self.move_to_breakeven = False  # Flag to check if SL moved to breakeven

    def _check_trade_exit(self):
        reward = 0
        current_df_index = self._current_tick - self.window_size + self.frame_bound[0]
        current_high = self.df['high'].iloc[current_df_index]
        current_low = self.df['low'].iloc[current_df_index]
        exit_price = None

        if self._trade_active:
            if self._position == Positions.Long:
                if not self.move_to_breakeven and current_high >= self.entry_price + 0.5:
                    self.stop_loss = self.entry_price  # Move SL to breakeven
                    self.move_to_breakeven = True
                if self.move_to_breakeven:
                    new_trailing_stop = max(self.stop_loss, current_high - 0.5)
                    self.stop_loss = new_trailing_stop  # Ensure SL only moves upwards
                if current_low <= self.stop_loss:
                    exit_price = self.stop_loss
            elif self._position == Positions.Short:
                if not self.move_to_breakeven and current_low <= self.entry_price - 0.5:
                    self.stop_loss = self.entry_price  # Move SL to breakeven
                    self.move_to_breakeven = True
                if self.move_to_breakeven:
                    new_trailing_stop = min(self.stop_loss, current_low + 0.5)
                    self.stop_loss = new_trailing_stop  # Ensure SL only moves downwards
                if current_high >= self.stop_loss:
                    exit_price = self.stop_loss

        if exit_price is not None:
            entry_price = self.entry_price
            if self._position == Positions.Long:
                reward += ((exit_price - entry_price) / entry_price) * 3000
            elif self._position == Positions.Short:
                reward += ((entry_price - exit_price) / entry_price) * 3000

            self._total_reward += reward
            self._total_profit += reward
            self._trade_active = False

        return reward



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

    def _get_observation(self):
        return self.signal_features[(self._current_tick-self.window_size+1):self._current_tick+1]

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

    def max_possible_profit(self):  # trade fees are ignored
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            current_price = self.prices[current_tick - 1]
            last_trade_price = self.prices[last_trade_tick]

            if self.unit_side == 'left':
                if position == Positions.Short:
                    quantity = profit * (last_trade_price - self.trade_fee)
                    profit = quantity / current_price

            elif self.unit_side == 'right':
                if position == Positions.Long:
                    quantity = profit / last_trade_price
                    profit = quantity * (current_price - self.trade_fee)

            last_trade_tick = current_tick - 1

        return profit