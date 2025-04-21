import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import warnings

class TradingBotSimulator:
    def __init__(self, m5_data, m15_data, h1_data, h4_data):
        self.m5_data = m5_data.copy()
        self.m15_data = m15_data.copy()
        self.h1_data = h1_data.copy()
        self.h4_data = h4_data.copy()

        self.tp_offset = 1.1
        self.sl_offset = 1.0
        self.entry_offset = 0.01

        # === PRECOMPUTE INDICATORS ===
        self.m5_data['ATR'] = self.calculate_atr(self.m5_data)
        self.m5_data['RSI'] = self.rsi_series(self.m5_data)

        self.m15_data['ema_5'] = self.m15_data['close'].ewm(span=5, adjust=False).mean()
        self.m15_data['ema_20'] = self.m15_data['close'].ewm(span=20).mean()

        self.h1_data['ema'] = self.h1_data['close'].ewm(span=20, adjust=False).mean()
        self.h4_data['ema'] = self.h4_data['close'].ewm(span=20, adjust=False).mean()

    def macro_trend(self, df_h1, df_h4):
        df_h1 = df_h1.copy()
        df_h4 = df_h4.copy()
        df_h1['ema'] = df_h1['close'].ewm(span=20, adjust=False).mean()
        df_h4['ema'] = df_h4['close'].ewm(span=20, adjust=False).mean()

        h1_trend = "UP" if df_h1['close'].iloc[-1] > df_h1['ema'].iloc[-1] else "DOWN"
        h4_trend = "UP" if df_h4['close'].iloc[-1] > df_h4['ema'].iloc[-1] else "DOWN"

        return h1_trend, h4_trend
    

    def rsi_series(self, df, period=14):
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_atr(self, df, period=14):
        df_copy = df.copy()
        df_copy['H-L'] = df_copy['high'] - df_copy['low']
        df_copy['H-C'] = abs(df_copy['high'] - df_copy['close'].shift())
        df_copy['L-C'] = abs(df_copy['low'] - df_copy['close'].shift())
        tr = df_copy[['H-L', 'H-C', 'L-C']].max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr


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
            is_valid_range(candle_1) and
            is_valid_range(candle_2) and
            is_valid_range(candle_3)
        )

        return candles


    def ema_short(self, df_15, period=5):
        df_15['ema'] = df_15['close'].ewm(span=period, adjust=False).mean()
        bullish = df_15['close'].iloc[-1] > df_15['ema'].iloc[-1]
        bearish = df_15['close'].iloc[-1] < df_15['ema'].iloc[-1]
        return bullish, bearish
    
    
    def ema_long(self, df_15):
        df_15['ema_long'] = df_15['close'].ewm(span=20).mean()
        uptrend = df_15['close'].iloc[-1] > df_15['ema_long'].iloc[-1]
        downtrend = df_15['close'].iloc[-1] < df_15['ema_long'].iloc[-1]

        return uptrend, downtrend

    def simulate(self):
        trades = []
        active_trade = None
        equity = 10  # Starting capital for tracking

        for i in range(20, len(self.m5_data)):
            current_candle = self.m5_data.iloc[i]
            current_time = current_candle['time']

            # Get sliced data up to this open time
            df_5 = self.m5_data[self.m5_data['time'] <= current_time].copy()
            df_15 = self.m15_data[self.m15_data['time'] <= current_time].copy()
            df_h1 = self.h1_data[self.h1_data['time'] <= current_time].copy()
            df_h4 = self.h4_data[self.h4_data['time'] <= current_time].copy()

            # Monitor active trade
            if active_trade:
                if active_trade['direction'] == "BUY":
                    if current_candle['low'] <= active_trade['sl']:
                        result = "SL"
                        reward = active_trade['sl'] - active_trade['entry']
                    elif current_candle['high'] >= active_trade['tp']:
                        result = "TP"
                        reward = active_trade['tp'] - active_trade['entry']
                    else:
                        continue  # Still active
                elif active_trade['direction'] == "SELL":
                    if current_candle['high'] >= active_trade['sl']:
                        result = "SL"
                        reward = active_trade['entry'] - active_trade['sl']
                    elif current_candle['low'] <= active_trade['tp']:
                        result = "TP"
                        reward = active_trade['entry'] - active_trade['tp']
                    else:
                        continue  # Still active

                active_trade['result'] = result
                active_trade['reward'] = reward
                trades.append(active_trade)
                equity += reward
                active_trade = None  # Reset to look for next trade
                continue  # After trade closes, wait until next candle

            # No active trade: check for new signal
            if current_time.hour in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
                # Compute indicators
                candles = self.candles_patern_5(df_5)
                atr_val = df_5['ATR'].iloc[-1]
                rsi_val = df_5['RSI'].iloc[-1]
                bullish_ema = df_15['close'].iloc[-1] > df_15['ema_5'].iloc[-1]
                bearish_ema = df_15['close'].iloc[-1] < df_15['ema_5'].iloc[-1]
                uptrend = df_15['close'].iloc[-1] > df_15['ema_20'].iloc[-1]
                downtrend = df_15['close'].iloc[-1] < df_15['ema_20'].iloc[-1]
                h1_trend, h4_trend = self.macro_trend(df_h1, df_h4)
                tp_val = atr_val * self.tp_offset
                sl_val = atr_val * self.sl_offset
                slippage = self.entry_offset * atr_val
                entry_price = current_candle['close']

                direction = None
                if candles and bullish_ema and uptrend and rsi_val < 50 and h1_trend == "UP" and h4_trend == "UP":
                    direction = "BUY"
                    entry = entry_price + slippage
                    tp = round(entry + tp_val, 2)
                    sl = round(entry - sl_val, 2)
                elif candles and bearish_ema and downtrend and rsi_val > 50 and h1_trend == "DOWN" and h4_trend == "DOWN":
                    direction = "SELL"
                    entry = entry_price - slippage
                    tp = round(entry - tp_val, 2)
                    sl = round(entry + sl_val, 2)

                if direction:
                    active_trade = {
                        "time": current_time,
                        "date": current_time.date(),
                        "hour": current_time.hour,
                        "candle_range": abs(current_candle['high'] - current_candle['low']),
                        "direction": direction,
                        "entry": entry,
                        "tp": tp,
                        "sl": sl,
                        "result": "NONE",
                        "reward": 0
                    }

        return pd.DataFrame(trades)



# Load historical data
m5_data = pd.read_csv("XAUUSD_M5.csv", parse_dates=['time'])
m15_data = pd.read_csv("XAUUSD_M15.csv", parse_dates=['time'])
h1_data = pd.read_csv("XAUUSD_H1.csv", parse_dates=['time'])
h4_data = pd.read_csv("XAUUSD_H4.csv", parse_dates=['time'])

# Run simulation
bot = TradingBotSimulator(m5_data, m15_data, h1_data, h4_data)
results = bot.simulate()

# Save or analyze
results.to_csv("simulation_results.csv", index=False)
print(results.head())

# Summary stats
total_trades = len(results)
tp_count = (results['result'] == 'TP').sum()
sl_count = (results['result'] == 'SL').sum()
completed_trades = tp_count + sl_count
win_rate = (tp_count / completed_trades) * 100 if completed_trades > 0 else 0

# loss, win, none hours stats
loss_hours = results[results['result'] == 'SL']['hour'].value_counts()
loss_hours = loss_hours.sort_index()
win_hours = results[results['result'] == 'TP']['hour'].value_counts()
win_hours = win_hours.sort_index()
none_hours = results[results['result'] == 'NONE']['hour'].value_counts()
none_hours = none_hours.sort_index()

# loss, win, none dates stats
loss_dates = results[results['result'] == 'SL']['date'].value_counts()
loss_dates = loss_dates.sort_index()
win_dates = results[results['result'] == 'TP']['date'].value_counts()
win_dates = win_dates.sort_index()
none_dates = results[results['result'] == 'NONE']['date'].value_counts()
none_dates = none_dates.sort_index()


average_candle_range_for_wins = results[results['result'] == 'TP']['candle_range'].mean()
average_candle_range_for_losses = results[results['result'] == 'SL']['candle_range'].mean()
average_candle_range_for_none = results[results['result'] == 'NONE']['candle_range'].mean()
print(f"Average Candle Range for Wins: {average_candle_range_for_wins:.2f}")
print(f"Average Candle Range for Losses: {average_candle_range_for_losses:.2f}")
print(f"Average Candle Range for None: {average_candle_range_for_none:.2f}")

print("\n--- Summary ---")
print(f"Total Trades: {total_trades}")
print(f"TP: {tp_count}")
print(f"SL: {sl_count}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Loss Hours:\n{loss_hours}")
print(f"Win Hours:\n{win_hours}")
print(f"None Hours:\n{none_hours}")

print("\n--- Daily Summary ---")
print(f"Total Trades: {total_trades}")
print(f"TP: {tp_count}")
print(f"SL: {sl_count}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Loss Dates:\n{loss_dates}")
print(f"Win Dates:\n{win_dates}")
print(f"None Dates:\n{none_dates}")

# Initialize variables for hourly profit and loss tracking
# Initialize hourly profit/loss tracking
hourly_profit = {}
hourly_loss = {}
date_profit = {}
date_loss = {}

# Total profit/loss
total_profit = 0
total_loss = 0

# Process each trade
for trade in results.itertuples():
    hour = trade.hour
    date = trade.date
    reward = trade.reward

    if hour not in hourly_profit:
        hourly_profit[hour] = 0
    if hour not in hourly_loss:
        hourly_loss[hour] = 0
    if date not in date_profit:
        date_profit[date] = 0
    if date not in date_loss:
        date_loss[date] = 0

    if reward > 0:
        hourly_profit[hour] += reward
        date_profit[date] += reward
        total_profit += reward
    elif reward < 0:
        hourly_loss[hour] += abs(reward)
        date_loss[date] += abs(reward)
        total_loss += abs(reward)

# Now calculate PLR per hour
hourly_plr_ratio = {}
dately_plr_ratio = {}
all_hours = results['hour'].unique()
all_dates = results['date'].unique()
for hour in sorted(all_hours):
    profit = hourly_profit.get(hour, 0)
    loss = hourly_loss.get(hour, 0)
    if loss > 0:
        hourly_plr_ratio[hour] = profit / loss
    else:
        hourly_plr_ratio[hour] = float('inf') if profit > 0 else 0

for date in sorted(all_dates):
    profit = date_profit.get(date, 0)
    loss = date_loss.get(date, 0)
    if loss > 0:
        dately_plr_ratio[date] = profit / loss
    else:
        dately_plr_ratio[date] = float('inf') if profit > 0 else 0

# Overall PLR
overall_plr = total_profit / total_loss if total_loss > 0 else float('inf')

# Print the hourly PLR and overall PLR
print(f"Hourly PLR:")
for hour, plr in sorted(hourly_plr_ratio.items()):
    print(f"Hour {hour}: {plr:.2f}")

print(f"\nDaily PLR:")
for date, plr in sorted(dately_plr_ratio.items()):
    print(f"Date {date}: {plr:.2f}")

returns = results['reward'][results['reward'] != 0]
if not returns.empty:
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)  # if returns ~ daily
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

print(f"\nOverall PLR: {overall_plr:.2f}")

# Equity curve
equity = 10  # Starting capital for tracking
balance = []
for trade in results.itertuples():
    if trade.reward not in [None, 0]:
        equity += trade.reward  # Add reward to equity

    balance.append(equity)

plt.plot(balance)
plt.title("Equity Curve")
plt.xlabel("Trade Number")
plt.ylabel("Account Balance ($)")
plt.grid(True)
plt.show()