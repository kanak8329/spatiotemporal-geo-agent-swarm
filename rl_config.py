"""Central configuration for the RL system."""

RL_CONFIG = {
    # Environment
    "target_gw_level": -15.0,       # Target sustainable GW level (meters)
    "max_episode_steps": 252,       # 21 years of monthly data
    "pumping_scale": 0.2,           # Action sensitivity (max 20% adjustment per step)

    # Reward weights
    "w_sustainability": 1.0,
    "w_extraction_efficiency": 0.3,
    "w_subsidence_penalty": 0.5,
    "w_action_volatility": 0.1,

    # Training
    "total_timesteps": 100000,
    "learning_rate": 3e-4,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,

    # Curriculum phases
    "curriculum": {
        "phase_a": {"pumping_range": (0, 200), "include_subsidence": False},
        "phase_b": {"pumping_range": (0, 500), "include_subsidence": False},
        "phase_c": {"pumping_range": (0, 1000), "include_subsidence": True},
    },

    # V&V thresholds
    "z_score_threshold": 4.0,
    "hitl_confidence_threshold": 0.7,
    "gw_jump_max_m_per_month": 20.0,

    # Checkpointing
    "checkpoint_freq": 10000,
    "eval_freq": 5000,
    "n_eval_episodes": 10,
}
