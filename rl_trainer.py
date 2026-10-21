import pickle
import pandas as pd
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from rich.console import Console
from rich.table import Table

from rl_environment import GroundwaterEnv
from validation_pipeline import ValidationPipeline
from rl_config import RL_CONFIG

console = Console()

class RichMetricsCallback(BaseCallback):
    """Log metrics to terminal with rich formatting."""
    def __init__(self, log_freq: int = 1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            if len(self.episode_rewards) > 0:
                table = Table(title=f"📊 Step {self.n_calls:,}")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")
                table.add_row("Mean Reward (last 100)", f"{np.mean(self.episode_rewards[-100:]):.3f}")
                table.add_row("Std Reward", f"{np.std(self.episode_rewards[-100:]):.3f}")
                table.add_row("Total Episodes", f"{len(self.episode_rewards):,}")
                console.print(table)
        return True

    def _on_rollout_end(self):
        if "episode" in self.locals.get("infos", [{}])[-1]:
            ep_info = self.locals["infos"][-1]["episode"]
            self.episode_rewards.append(ep_info["r"])

def train_rl_agent(resume_from: str = None):
    console.print("[bold cyan]🧠 RL Training Pipeline Starting...[/]")
    
    os.makedirs("./rl_checkpoints", exist_ok=True)
    os.makedirs("./rl_tensorboard", exist_ok=True)

    # 1. Load world model
    console.print("[yellow]Loading XGBoost world model...[/]")
    try:
        with open("xgboost_groundwater_model.pkl", "rb") as f:
            world_model = pickle.load(f)
    except FileNotFoundError:
        console.print("[bold red]Error: xgboost_groundwater_model.pkl not found. Run Phase 5 first![/]")
        return

    # 2. Load dataset
    console.print("[yellow]Loading dataset for environment states from V&V vault...[/]")
    df = pd.read_parquet("./data/validated")
    
    console.print("[yellow]Engineering temporal features...[/]")
    df = df.sort_values(by=['longitude', 'latitude', 'timestamp'])
    df['lag_1_gw'] = df.groupby(['longitude', 'latitude'])['groundwater_level'].shift(1)
    df['rolling_mean_3_gw'] = df.groupby(['longitude', 'latitude'])['groundwater_level'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['lag_1_gw'] = df['lag_1_gw'].bfill()
    df['rolling_mean_3_gw'] = df['rolling_mean_3_gw'].bfill()
    
    # Optional: Quick validation (to ensure data is clean)
    console.print("[yellow]Running quick V&V check...[/]")
    pipeline = ValidationPipeline()
    df, vv_results = pipeline.run(df)
    
    for r in vv_results:
        if r.severity.value in ["critical", "blocker"] and not r.passed:
             console.print(f"[bold red]❌ {r.check_name}: {r.message}[/]")

    # 3. Create environment
    console.print("[yellow]Initializing GroundwaterEnv...[/]")
    env = Monitor(GroundwaterEnv(world_model, df, RL_CONFIG))
    eval_env = Monitor(GroundwaterEnv(world_model, df, RL_CONFIG))

    # 4. Create or resume agent
    if resume_from and os.path.exists(resume_from):
        console.print(f"[bold green]Resuming from checkpoint: {resume_from}[/]")
        model = PPO.load(resume_from, env=env)
    else:
        console.print("[bold green]Creating new PPO Agent...[/]")
        model = PPO(
            "MlpPolicy", env,
            learning_rate=RL_CONFIG["learning_rate"],
            n_steps=2048,
            batch_size=RL_CONFIG["batch_size"],
            n_epochs=RL_CONFIG["n_epochs"],
            gamma=RL_CONFIG["gamma"],
            gae_lambda=RL_CONFIG["gae_lambda"],
            clip_range=RL_CONFIG["clip_range"],
            ent_coef=RL_CONFIG["ent_coef"],
            verbose=0,
            tensorboard_log="./rl_tensorboard",
            device="auto", 
        )

    # 5. Setup callbacks
    callbacks = [
        CheckpointCallback(save_freq=RL_CONFIG["checkpoint_freq"], save_path="./rl_checkpoints", name_prefix="gw_rl"),
        EvalCallback(eval_env, best_model_save_path="./rl_checkpoints/best",
                     log_path="./rl_checkpoints/best", eval_freq=RL_CONFIG["eval_freq"], 
                     n_eval_episodes=RL_CONFIG["n_eval_episodes"], deterministic=True, render=False),
        RichMetricsCallback(log_freq=2000),
    ]

    # 6. Train
    total_timesteps = RL_CONFIG["total_timesteps"]
    console.print(f"\n[bold green]🚀 Training for {total_timesteps:,} timesteps...[/]")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name="groundwater_ppo",
        reset_num_timesteps=False
    )

    # 7. Save final policy
    model.save("rl_groundwater_policy")
    console.print("\n[bold green]✅ RL Policy saved to rl_groundwater_policy.zip[/]")
    
    return model

if __name__ == "__main__":
    train_rl_agent()
