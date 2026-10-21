import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd
from rl_config import RL_CONFIG

class GroundwaterEnv(gym.Env):
    """
    RL environment for groundwater pumping policy optimization.
    Uses XGBoost as the dynamics model (world model).
    """
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, world_model, dataset, config=None):
        super().__init__()
        self.world_model = world_model   # Loaded XGBoost model
        self.dataset = dataset           # Validated parquet DataFrame
        self.config = config or RL_CONFIG

        # State: 10 environmental features + 2 derived (season_sin, season_cos)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        # Action: pumping adjustment [-1.0, +1.0]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.current_step = 0
        self.current_location_idx = 0
        self.target_gw_level = self.config.get("target_gw_level", -15.0)
        self.max_steps = self.config.get("max_episode_steps", 252)
        
        # Group data by location for episodes
        self.locations = self.dataset.groupby(["longitude", "latitude"])
        self.loc_keys = list(self.locations.groups.keys())
        self.trajectory = pd.DataFrame()
        self.current_state_features = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Sample a random spatial location's time series
        self.current_location_idx = self.np_random.integers(0, len(self.loc_keys))
        loc = self.loc_keys[self.current_location_idx]
        
        self.trajectory = self.locations.get_group(loc).sort_values("timestamp").reset_index(drop=True)
        self.current_step = 0
        
        # Initialize current state
        row = self.trajectory.iloc[self.current_step]
        self.current_state_features = {
            'rainfall': row['rainfall'],
            'temperature': row['temperature'],
            'pumping': row['pumping'],
            'lulc': row['lulc'],
            'subsidence': row['subsidence'],
            'elevation_dem': row['elevation_dem'],
            'twi': row['twi'],
            'distance_to_river': row['distance_to_river'],
            'lag_1_gw': row['lag_1_gw'],
            'rolling_mean_3_gw': row['rolling_mean_3_gw'],
            'groundwater_level': row['groundwater_level']
        }
        
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.trajectory.iloc[self.current_step]
        month = row["timestamp"].month
        
        obs = np.array([
            self.current_state_features['groundwater_level'], 
            self.current_state_features['rainfall'], 
            self.current_state_features['temperature'],
            self.current_state_features['pumping'], 
            self.current_state_features['lulc'], 
            self.current_state_features['subsidence'],
            self.current_state_features['elevation_dem'], 
            self.current_state_features['twi'], 
            self.current_state_features['distance_to_river'],
            np.sin(2 * np.pi * month / 12),  # season_sin
            np.cos(2 * np.pi * month / 12),  # season_cos
            self._get_delta_gw(),
        ], dtype=np.float32)
        return obs

    def _get_delta_gw(self):
        if self.current_step == 0:
            return 0.0
        return self.current_state_features['groundwater_level'] - self.current_state_features['lag_1_gw']

    def step(self, action):
        pumping_adj = float(action[0])
        row = self.trajectory.iloc[self.current_step]

        # Apply action: modify pumping rate from the real dataset by up to +/- 20%
        base_pumping = row["pumping"]
        pumping_scale = self.config.get("pumping_scale", 0.2)
        new_pumping = np.clip(base_pumping * (1 + pumping_adj * pumping_scale), 0, 5000)
        
        # Update current state for prediction
        self.current_state_features['pumping'] = new_pumping

        # Features to feed into XGBoost
        model_features = [
            'rainfall', 'temperature', 'pumping', 'lulc', 'subsidence',
            'elevation_dem', 'twi', 'distance_to_river',
            'lag_1_gw', 'rolling_mean_3_gw'
        ]
        
        pred_df = pd.DataFrame([self.current_state_features])[model_features].astype(float)
        
        # Predict the next groundwater level using XGBoost
        pred_gw = self.world_model.predict(pred_df)[0]

        # Compute reward
        w_sus = self.config.get("w_sustainability", 1.0)
        w_sub = self.config.get("w_subsidence_penalty", 0.5)
        w_vol = self.config.get("w_action_volatility", 0.1)
        
        sustainability_score = -abs(pred_gw - self.target_gw_level)
        subsidence_pen = max(0, self.current_state_features["subsidence"] - 50) ** 2 * 0.001
        volatility_pen = abs(pumping_adj) * 0.1
        
        reward = (w_sus * sustainability_score) - (w_sub * subsidence_pen) - (w_vol * volatility_pen)

        self.current_step += 1
        terminated = self.current_step >= len(self.trajectory) - 1 or self.current_step >= self.max_steps
        truncated = False

        # Transition state to next step
        if not terminated:
            next_row = self.trajectory.iloc[self.current_step]
            # Real physical features advance
            self.current_state_features['rainfall'] = next_row['rainfall']
            self.current_state_features['temperature'] = next_row['temperature']
            self.current_state_features['lulc'] = next_row['lulc']
            self.current_state_features['subsidence'] = next_row['subsidence']
            
            # Autoregressive updates based on our prediction
            self.current_state_features['lag_1_gw'] = self.current_state_features['groundwater_level']
            self.current_state_features['groundwater_level'] = pred_gw
            
            # Simple rolling mean approximation
            self.current_state_features['rolling_mean_3_gw'] = (
                self.current_state_features['groundwater_level'] + 
                self.current_state_features['lag_1_gw'] + 
                next_row['lag_1_gw']  # proxy for lag 2
            ) / 3.0

        return self._get_obs(), float(reward), terminated, truncated, {
            "predicted_gw": pred_gw,
            "pumping_applied": new_pumping,
            "sustainability_score": sustainability_score,
            "reward": float(reward)
        }
