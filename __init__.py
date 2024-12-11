# __init__.py
from .trading_env import TradingEnv, Actions, Positions
from .forex_env import ForexEnv
from gym.envs.registration import register

# Register your custom environment
register(
    id='forex-v0',
    entry_point='your_module.ForexEnv',  # Replace 'your_module' with the correct module path
    kwargs={
        'df': None,  # Placeholder; you will pass this when creating the environment
        'window_size': 10,
        'frame_bound': (10, 100),
        'unit_side': 'left',
    },
)