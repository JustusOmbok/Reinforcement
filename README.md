# Forex Trading Bot with RecurrentPPO and Optuna

This repository implements a Forex trading bot that uses MetaTrader5 for data fetching, Stable-Baselines3's RecurrentPPO for reinforcement learning, and Optuna for hyperparameter optimization. The goal is to create an automated trading system capable of learning and optimizing its strategy using historical and real-time Forex data.

## Features

MetaTrader5 Integration: Fetches historical and real-time Forex data.

Reinforcement Learning: Utilizes RecurrentPPO (from Stable-Baselines3) for decision-making.

Custom Gym Environment: Includes a custom Forex trading environment compatible with Stable-Baselines3.

Technical Indicators: Implements a variety of indicators for feature engineering.

Hyperparameter Optimization: Uses Optuna to optimize model parameters for better performance.

Data Normalization: Applies MinMaxScaler for consistent input ranges.

## Requirements

### Libraries

Install the required Python libraries using the following command:

pip install MetaTrader5 pandas numpy matplotlib stable-baselines3 sb3-contrib gymnasium optuna scikit-learn gym-anytrading finta

MetaTrader5

Ensure that MetaTrader5 is installed and set up on your machine. Update your account credentials in the script.

File Structure

fetch_data: Fetches and preprocesses historical data from MetaTrader5.

create_features: Generates technical indicators used as features.

normalize_indicators: Normalizes feature values for the model.

CustomForexEnv: Defines a custom Gym environment for trading.

main_trading_loop: Runs the training and evaluation process.

tune_hyperparameters: Uses Optuna to find the best hyperparameters for the model.

### Usage

1. MetaTrader5 Initialization

Replace the placeholders with your MetaTrader5 credentials:

if not mt5.initialize(login=100003703, server="FBS-Demo", password="jV9}8d,)"):
    print("Failed to initialize MetaTrader5")
    mt5.shutdown()
    quit()

2. Fetch Data

Historical Forex data is fetched using:

df = fetch_data("XAUUSD", mt5.TIMEFRAME_D1, datetime.datetime(2024, 1, 1), datetime.datetime.now())

Ensure the symbol and timeframe are correct for your requirements.

3. Feature Engineering

Generate technical indicators and normalize features:

df = create_features(df)
df.fillna(0, inplace=True)

4. Hyperparameter Tuning

Optimize the RecurrentPPO model parameters using Optuna:

best_params = tune_hyperparameters()
print(f"Optimized Parameters: {best_params}")

5. Training and Evaluation

Train the RecurrentPPO model using the optimal parameters and evaluate its performance:

model.learn(total_timesteps=500000)
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5, deterministic=True)
print(f"Mean Reward: {mean_reward}")

### Optuna Integration

Hyperparameter Optimization

The Optuna objective function samples hyperparameters, trains the model, and evaluates its performance:

def objective(trial):
    n_steps = trial.suggest_int("n_steps", 16, 2048, log=True)
    gamma = trial.suggest_float("gamma", 0.8, 0.9999, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 0.1, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 1.0)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.5, 5.0)

    # Environment and model setup
    env_maker = lambda: CustomForexEnv(df=df, window_size=12, frame_bound=(12, len(df)))
    env = VecNormalize(DummyVecEnv([env_maker]), norm_obs=True, norm_reward=True)

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        n_steps=n_steps,
        gamma=gamma,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        gae_lambda=gae_lambda,
        max_grad_norm=max_grad_norm,
        verbose=0,
    )

    model.learn(total_timesteps=100000)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5, deterministic=True)

    return mean_reward

#### Run the optimization process:

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print(f"Best Parameters: {study.best_params}")

#### Example Output

After tuning hyperparameters with Optuna:

Best Parameters: {'n_steps': 512, 'gamma': 0.99, 'learning_rate': 0.0005, 'ent_coef': 0.005, 'vf_coef': 0.5, 'gae_lambda': 0.95, 'max_grad_norm': 2.0}
Best Mean Reward: 120.75

#### Future Improvements

Dynamic Symbol and Timeframe: Allow user input for the trading symbol and timeframe.

Additional Indicators: Add more indicators or use feature selection to improve performance.

Real-Time Trading: Integrate real-time data feeds and execution.

Performance Visualization: Plot model predictions versus actual price movements.

#### Troubleshooting

Common Issues

MetaTrader5 Initialization Failed: Verify your login credentials and ensure MetaTrader5 is running.

Empty Dataframe: Check the symbol, timeframe, and date range.

Model Training Errors: Ensure all input features are normalized and have no missing values.

Logging

Enable logging for detailed insights:

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Starting hyperparameter tuning...")

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

MetaTrader5 for data access and trading.

Stable-Baselines3 for reinforcement learning.

Optuna for hyperparameter optimization.

Finta for technical indicators.

Gym-AnyTrading for trading environments.

For questions or support, feel free to reach out!