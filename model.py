import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Categorical
from datetime import datetime
import matplotlib.pyplot as plt
from collections import deque

# --- CNN-GRU A3C Shared Model ---
class CNN_GRU_A3CNet(nn.Module):
    def __init__(self, input_channels, sequence_length, action_space):
        super(CNN_GRU_A3CNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.gru = nn.GRU(input_size=64, hidden_size=64, batch_first=True)  # 🔧 Fix applied
        self.fc = nn.Linear(64, 128)
        self.policy_head = nn.Linear(128, 3)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, h_n = self.gru(x)
        x = torch.relu(self.fc(h_n[-1]))
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

# --- Trading Environment ---
class TradingEnv:
    def __init__(self, data):
        self.data = data.reset_index(drop=True)
        self.current_step = 0
        self.done = False
        self.position = 0
        self.entry_price = 0
        self.initial_balance = 10
        self.balance = self.initial_balance
        self.total_reward = 0
        self.results = []

    def reset(self):
        self.current_step = 0
        self.done = False
        self.position = 0
        self.entry_price = 0
        self.balance = self.initial_balance
        self.total_reward = 0
        self.results = []
        return self._get_state()

    def _get_state(self):
        window = self.data.iloc[self.current_step:self.current_step+10][['rsi', 'ema_short', 'ema_long', 'atr', 'candle_range']]
        return window.values.T.astype(np.float32)

    def step(self, action):
        row = self.data.iloc[self.current_step + 9]
        reward = 0
        result = 'NONE'

        if action == 1:
            if self.position == 0:
                self.entry_price = row['close']
                self.position = 1
            elif self.position == -1:
                reward = self.entry_price - row['close']
                self.balance += reward
                self.total_reward += reward
                self.position = 0
                result = 'TP' if reward > 0 else 'SL'

        elif action == 2:
            if self.position == 0:
                self.entry_price = row['close']
                self.position = -1
            elif self.position == 1:
                reward = row['close'] - self.entry_price
                self.balance += reward
                self.total_reward += reward
                self.position = 0
                result = 'TP' if reward > 0 else 'SL'

        self.results.append({
            'hour': row['time'].hour,
            'reward': reward,
            'result': result,
            'candle_range': row['candle_range']
        })

        self.current_step += 1
        if self.current_step + 10 >= len(self.data):
            self.done = True

        return self._get_state(), reward, self.done, {}

# --- A3C Worker Process ---
def worker(global_model, optimizer, data, gamma=0.99, max_eps=10):
    local_model = CNN_GRU_A3CNet(5, 10, 3)
    env = TradingEnv(data)

    for ep in range(max_eps):
        state = env.reset()
        log_probs = []
        values = []
        rewards = []

        while not env.done:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            policy_logits, value = local_model(state_tensor)

            probs = torch.softmax(policy_logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()

            next_state, reward, done, _ = env.step(action.item())

            log_probs.append(dist.log_prob(action))
            values.append(value.squeeze(0))
            rewards.append(torch.tensor([reward], dtype=torch.float))

            state = next_state

        Qval = torch.tensor([0.0]) if env.done else local_model(torch.from_numpy(state).float().unsqueeze(0))[1].detach()
        returns = []
        for r in reversed(rewards):
            Qval = r + gamma * Qval
            returns.insert(0, Qval)

        returns = torch.cat(returns)
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)
        advantage = returns - values

        policy_loss = -(log_probs * advantage.detach()).mean()
        value_loss = advantage.pow(2).mean()
        loss = policy_loss + 0.5 * value_loss

        optimizer.zero_grad()
        loss.backward()
        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            global_param._grad = local_param.grad
        optimizer.step()
        local_model.load_state_dict(global_model.state_dict())

    # Summary stats
    results = pd.DataFrame(env.results)
    total_trades = len(results)
    tp_count = (results['result'] == 'TP').sum()
    sl_count = (results['result'] == 'SL').sum()
    completed_trades = tp_count + sl_count
    win_rate = (tp_count / completed_trades) * 100 if completed_trades > 0 else 0

    loss_hours = results[results['result'] == 'SL']['hour'].value_counts().sort_index()
    win_hours = results[results['result'] == 'TP']['hour'].value_counts().sort_index()
    none_hours = results[results['result'] == 'NONE']['hour'].value_counts().sort_index()

    avg_win_range = results[results['result'] == 'TP']['candle_range'].mean()
    avg_loss_range = results[results['result'] == 'SL']['candle_range'].mean()
    avg_none_range = results[results['result'] == 'NONE']['candle_range'].mean()

    print(f"Average Candle Range for Wins: {avg_win_range:.2f}")
    print(f"Average Candle Range for Losses: {avg_loss_range:.2f}")
    print(f"Average Candle Range for None: {avg_none_range:.2f}")

    print("\n--- Summary ---")
    print(f"Total Trades: {total_trades}")
    print(f"TP: {tp_count}")
    print(f"SL: {sl_count}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Loss Hours:\n{loss_hours}")
    print(f"Win Hours:\n{win_hours}")
    print(f"None Hours:\n{none_hours}")

    hourly_profit = {}
    hourly_loss = {}
    total_profit = 0
    total_loss = 0

    for trade in results.itertuples():
        hour = trade.hour
        reward = trade.reward

        if hour not in hourly_profit:
            hourly_profit[hour] = 0
        if hour not in hourly_loss:
            hourly_loss[hour] = 0

        if reward > 0:
            hourly_profit[hour] += reward
            total_profit += reward
        elif reward < 0:
            hourly_loss[hour] += abs(reward)
            total_loss += abs(reward)

    hourly_plr_ratio = {}
    all_hours = results['hour'].unique()
    for hour in sorted(all_hours):
        profit = hourly_profit.get(hour, 0)
        loss = hourly_loss.get(hour, 0)
        hourly_plr_ratio[hour] = profit / loss if loss > 0 else float('inf') if profit > 0 else 0

    overall_plr = total_profit / total_loss if total_loss > 0 else float('inf')

    print("Hourly PLR:")
    for hour, plr in sorted(hourly_plr_ratio.items()):
        print(f"Hour {hour}: {plr:.2f}")

    print(f"\nOverall PLR: {overall_plr:.2f}")

    equity = 10
    balance = []
    for trade in results.itertuples():
        if trade.reward not in [None, 0]:
            equity += trade.reward
        balance.append(equity)

    plt.plot(balance)
    plt.title("Equity Curve")
    plt.xlabel("Trade Number")
    plt.ylabel("Account Balance ($)")
    plt.grid(True)
    plt.show()

# --- Prepare DataFrame Features ---
def prepare_features(m5_data, m15_data):
    m15_data['rsi'] = m15_data['close'].diff().apply(lambda x: max(x, 0)).rolling(14).mean() / \
                      (m15_data['close'].diff().abs().rolling(14).mean()) * 100
    m15_data['atr'] = m15_data['high'].combine(m15_data['low'], max) - m15_data['low']
    m15_data['ema_short'] = m15_data['close'].ewm(span=5).mean()
    m15_data['ema_long'] = m15_data['close'].ewm(span=20).mean()
    m15_data['candle_range'] = m15_data['high'] - m15_data['low']
    return m15_data.dropna().reset_index(drop=True)

# --- Main ---
if __name__ == "__main__":
    m5_data = pd.read_csv("XAUUSD_M5.csv", parse_dates=['time'])
    m15_data = pd.read_csv("XAUUSD_M15.csv", parse_dates=['time'])
    prepared_data = prepare_features(m5_data, m15_data)

    global_model = CNN_GRU_A3CNet(input_channels=5, sequence_length=10, action_space=3)
    global_model.share_memory()
    optimizer = optim.Adam(global_model.parameters(), lr=1e-4)

    processes = []
    for _ in range(mp.cpu_count()):
        p = mp.Process(target=worker, args=(global_model, optimizer, prepared_data))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("Training complete.")
