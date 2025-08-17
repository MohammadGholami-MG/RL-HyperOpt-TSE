import numpy as np
import xgboost as xgb
import lightgbm as lgb
import gymnasium as gym
import hashlib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from stable_baselines3 import PPO
from gymnasium import spaces



class RandomForestHyperparamEnv(gym.Env):
    def __init__(self, x_train, y_train, x_test, y_test):
        """
        Initialize the custom Gym environment for optimizing RandomForest hyperparameters.

        Parameters:
            x_train (ndarray): Training feature set
            y_train (ndarray): Training labels
            x_test (ndarray): Test feature set
            y_test (ndarray): Test labels
        """
        super(RandomForestHyperparamEnv, self).__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        # Actions: [n_estimators_scaled, max_depth_scaled] ∈ [0, 1]^2
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Dummy observation space (not used)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # Dictionary for caching model evaluation results for specific hyperparameter combinations
        self.cache = {}

    def reset(self, seed=None, options=None):
        """
        Reset the environment (dummy, one-step episode).
        """
        super().reset(seed=seed)
        self.current_obs = np.array([0.5], dtype=np.float32)
        return self.current_obs, {}

    def step(self, action):
        """
        Perform a step in the environment by evaluating the hyperparameters
        corresponding to the given action.

        Parameters:
            action (array): Normalized hyperparameters in [0, 1]

        Returns:
            observation (array), reward (float), terminated (bool), truncated (bool), info (dict)
        """
        # Convert scaled actions to actual hyperparameter values
        n_estimators = int(10 + action[0] * 90)     # Range: 10 to 100
        max_depth = int(2 + action[1] * 18)         # Range: 2 to 20

        # Create a unique cache key using a hash of the parameters
        key_str = f"{n_estimators}_{max_depth}"
        key_hash = hashlib.md5(key_str.encode()).hexdigest()

        if key_hash in self.cache:
            # Use cached result if already evaluated
            mse = self.cache[key_hash]
        else:
            # Train and evaluate the model
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
            model.fit(self.x_train, self.y_train)
            y_pred = model.predict(self.x_test)
            mse = mean_squared_error(self.y_test, y_pred)
            self.cache[key_hash] = mse  # Store result in cache

        reward = -mse  # The lower the MSE, the higher the reward

        terminated = True  # One-step environment
        truncated = False
        info = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'mse': mse
        }

        return self.current_obs, reward, terminated, truncated, info

def optimize_rf_with_rl(processed_data, timesteps=20000):
    """
    Optimize RandomForest hyperparameters (n_estimators and max_depth)
    for each (symbol, feature) combination using PPO reinforcement learning.

    Parameters:
        processed_data (dict): Dictionary containing training and testing splits 
                               for each (symbol, feature).
        timesteps (int): Number of training steps for PPO agent.

    Returns:
        dict: Best hyperparameters and MSE for each (symbol, feature) pair.
    """
    best_params = {}

    for (symbol, feature), data in processed_data.items():
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_test = data["x_test"]
        y_test = data["y_test"]

        print(f"\n... Optimizing RandomForest for {symbol} - {feature}")

        env = RandomForestHyperparamEnv(x_train, y_train, x_test, y_test)
        model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0003, n_steps=2048)
        model.learn(total_timesteps=timesteps)

        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        best_params[(symbol, feature)] = {
            "best_n_estimators": info['n_estimators'],
            "best_max_depth": info['max_depth'],
            "best_mse": info['mse']
        }

    return best_params


class XGBHyperparamEnv(gym.Env):
    def __init__(self, x_train, y_train, x_test, y_test):
        """
        Initialize the custom Gym environment for optimizing XGBoost hyperparameters.

        Parameters:
            x_train (ndarray): Training feature set
            y_train (ndarray): Training labels
            x_test (ndarray): Test feature set
            y_test (ndarray): Test labels
        """
        super(XGBHyperparamEnv, self).__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        # Action space: [n_estimators, max_depth, learning_rate] ∈ [0, 1]^3
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation space (dummy, not used in this stateless setup)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # Dictionary to cache evaluation results for hyperparameter combinations
        self.cache = {}

    def reset(self, seed=None, options=None):
        """
        Reset the environment (dummy, one-step episode).
        """
        super().reset(seed=seed)
        self.current_obs = np.array([0.5], dtype=np.float32)
        return self.current_obs, {}

    def step(self, action):
        """
        Perform a step in the environment by evaluating the hyperparameters
        corresponding to the given action.

        Parameters:
            action (array): Normalized hyperparameters in [0, 1]

        Returns:
            observation (array), reward (float), terminated (bool), truncated (bool), info (dict)
        """
        # Convert normalized actions to real hyperparameter values
        n_estimators = int(50 + action[0] * 150)         # Range: 50 to 200
        max_depth = int(2 + action[1] * 13)              # Range: 2 to 15
        learning_rate = 0.01 + action[2] * 0.29          # Range: 0.01 to 0.3

        # Generate a unique cache key using hashing
        key_str = f"{n_estimators}_{max_depth}_{learning_rate:.4f}"
        key_hash = hashlib.md5(key_str.encode()).hexdigest()

        if key_hash in self.cache:
            # Use cached MSE if this config was already evaluated
            mse = self.cache[key_hash]
        else:
            # Train XGBoost model on full training data
            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            model.fit(self.x_train, self.y_train)
            y_pred = model.predict(self.x_test)
            mse = mean_squared_error(self.y_test, y_pred)
            self.cache[key_hash] = mse  # Save to cache

        reward = -mse  # Lower MSE = higher reward

        terminated = True  # One-step environment
        truncated = False
        info = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'mse': mse
        }

        return self.current_obs, reward, terminated, truncated, info

def optimize_xgb_with_rl(processed_data, timesteps=20000):
    """
    Apply PPO reinforcement learning to find optimal hyperparameters
    for XGBoost on each (symbol, feature) data subset.

    Parameters:
        processed_data (dict): Dictionary with (symbol, feature) as keys and 
                               train/test splits as values.
        timesteps (int): Number of PPO training steps.

    Returns:
        dict: Best hyperparameters and corresponding MSEs for each subset.
    """
    best_params = {}

    for (symbol, feature), data in processed_data.items():
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_test = data["x_test"]
        y_test = data["y_test"]

        print(f"\n... Optimizing XGBoost for {symbol} - {feature}")

        env = XGBHyperparamEnv(x_train, y_train, x_test, y_test)
        model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0003, n_steps=2048)
        model.learn(total_timesteps=timesteps)

        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        best_params[(symbol, feature)] = {
            "best_n_estimators": info['n_estimators'],
            "best_max_depth": info['max_depth'],
            "best_learning_rate": info['learning_rate'],
            "best_mse": info['mse']
        }

    return best_params


class GBTHyperparamEnv(gym.Env):
    def __init__(self, x_train, y_train, x_test, y_test):
        """
        Initialize the custom Gym environment for optimizing 
        GradientBoostingRegressor hyperparameters.

        Parameters:
            x_train (ndarray): Training features
            y_train (ndarray): Training labels
            x_test (ndarray): Test features
            y_test (ndarray): Test labels
        """
        super(GBTHyperparamEnv, self).__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        # Action space: [n_estimators, max_depth, learning_rate] ∈ [0, 1]^3
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)

        # Dummy observation space (stateless)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # Caching evaluation results to avoid redundant computation
        self.cache = {}

    def reset(self, seed=None, options=None):
        """
        Reset the environment (one-step stateless setup).
        """
        super().reset(seed=seed)
        self.current_obs = np.array([0.5], dtype=np.float32)
        return self.current_obs, {}

    def step(self, action):
        """
        Execute one step of the environment by converting the action into
        real hyperparameters and evaluating the corresponding model.

        Parameters:
            action (array): Normalized action values for hyperparameters.

        Returns:
            observation (array), reward (float), terminated (bool),
            truncated (bool), info (dict)
        """
        # Convert normalized values to actual hyperparameters
        n_estimators = int(50 + action[0] * 150)         # Range: 50 to 200
        max_depth = int(2 + action[1] * 13)              # Range: 2 to 15
        learning_rate = 0.01 + action[2] * 0.29          # Range: 0.01 to 0.3

        # Create a unique hash key for caching
        key_str = f"{n_estimators}_{max_depth}_{learning_rate:.4f}"
        key_hash = hashlib.md5(key_str.encode()).hexdigest()

        if key_hash in self.cache:
            # Use cached result if available
            mse = self.cache[key_hash]
        else:
            # Train GBT model on full training data
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42
            )
            model.fit(self.x_train, self.y_train)  # Use full training data
            y_pred = model.predict(self.x_test)
            mse = mean_squared_error(self.y_test, y_pred)

            # Store in cache
            self.cache[key_hash] = mse

        reward = -mse  # Lower MSE gives higher reward

        terminated = True  # One-step episode
        truncated = False
        info = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'mse': mse
        }

        return self.current_obs, reward, terminated, truncated, info

def optimize_gbt_with_rl(processed_data, timesteps=20000):
    """
    Apply PPO to find the best hyperparameters for GradientBoostingRegressor
    across multiple (symbol, feature) combinations.

    Parameters:
        processed_data (dict): Dictionary with (symbol, feature) as keys and 
                               train/test data as values.
        timesteps (int): Total PPO training timesteps.

    Returns:
        dict: Dictionary of best hyperparameters and MSE for each combination.
    """
    best_params = {}

    for (symbol, feature), data in processed_data.items():
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_test = data["x_test"]
        y_test = data["y_test"]

        print(f"\n... Optimizing GradientBoosting for {symbol} - {feature}")

        env = GBTHyperparamEnv(x_train, y_train, x_test, y_test)
        model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0003, n_steps=2048)
        model.learn(total_timesteps=timesteps)

        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        best_params[(symbol, feature)] = {
            "best_n_estimators": info['n_estimators'],
            "best_max_depth": info['max_depth'],
            "best_learning_rate": info['learning_rate'],
            "best_mse": info['mse']
        }

    return best_params


class LightGBMHyperparamEnv(gym.Env):
    def __init__(self, x_train, y_train, x_test, y_test):
        """
        Initialize the custom Gym environment for optimizing 
        LightGBM hyperparameters.

        Parameters:
            x_train (ndarray): Training features
            y_train (ndarray): Training labels
            x_test (ndarray): Test features
            y_test (ndarray): Test labels
        """
        super(LightGBMHyperparamEnv, self).__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        # Action space: [num_leaves, max_depth, learning_rate] ∈ [0, 1]^3
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)

        # Dummy observation (stateless environment)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # Cache to store previously evaluated configurations
        self.cache = {}

    def reset(self, seed=None, options=None):
        """
        Reset the environment (stateless, one-step interaction).
        """
        super().reset(seed=seed)
        self.current_obs = np.array([0.5], dtype=np.float32)
        return self.current_obs, {}

    def step(self, action):
        """
        Apply one step using the action to evaluate LightGBM performance.

        Parameters:
            action (array): Normalized values for hyperparameters.

        Returns:
            observation (array), reward (float), terminated (bool),
            truncated (bool), info (dict)
        """
        # Convert normalized action values to real hyperparameters
        num_leaves = int(10 + action[0] * 290)          # Range: 10 to 300
        max_depth = int(3 + action[1] * 37)             # Range: 3 to 40
        learning_rate = 0.005 + action[2] * 0.395       # Range: 0.005 to 0.4

        # Generate hash key for caching
        key_str = f"{num_leaves}_{max_depth}_{learning_rate:.4f}"
        key_hash = hashlib.md5(key_str.encode()).hexdigest()

        if key_hash in self.cache:
            mse = self.cache[key_hash]
        else:
            # Train LightGBM model on full training data using CPU
            model = lgb.LGBMRegressor(
                num_leaves=num_leaves,
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            model.fit(self.x_train, self.y_train)  # Use full training data
            y_pred = model.predict(self.x_test)
            mse = mean_squared_error(self.y_test, y_pred)

            # Save to cache
            self.cache[key_hash] = mse

        reward = -mse  # The goal is to minimize MSE

        terminated = True  # One-step episode
        truncated = False
        info = {
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'mse': mse
        }

        return self.current_obs, reward, terminated, truncated, info

def optimize_lgb_with_rl(processed_data, timesteps=20000):
    """
    Use PPO to find the best hyperparameters for LightGBM 
    on each (symbol, feature) dataset.

    Parameters:
        processed_data (dict): Dictionary with (symbol, feature) as keys and 
                               train/test data as values.
        timesteps (int): PPO training timesteps.

    Returns:
        dict: Dictionary containing optimal hyperparameters and MSE values.
    """
    best_params = {}

    for (symbol, feature), data in processed_data.items():
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_test = data["x_test"]
        y_test = data["y_test"]

        print(f"\n... Optimizing LightGBM for {symbol} - {feature}")

        env = LightGBMHyperparamEnv(x_train, y_train, x_test, y_test)
        model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0003, n_steps=2048)
        model.learn(total_timesteps=timesteps)

        # Evaluate the final policy
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        best_params[(symbol, feature)] = {
            "best_num_leaves": info['num_leaves'],
            "best_max_depth": info['max_depth'],
            "best_learning_rate": info['learning_rate'],
            "best_mse": info['mse']
        }

    return best_params

