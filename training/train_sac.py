import os
import torch
import numpy as np

torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"

from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from curriculum import CurriculumCallback
from envs.drone_formation_env import DroneFormationEnv


class MetricsCallback(BaseCallback):
    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq != 0:
            return True

        infos = self.locals.get("infos")
        rewards = self.locals.get("rewards")

        if rewards is not None:
            self.logger.record("custom/step_reward_mean", float(np.mean(rewards)))

        if infos is not None and len(infos) > 0:
            keys = [
                "dist_target",
                "form_error",
                "tilt",
                "energy",
                "inter_drone_dist",
                "goal_reached",
                "collided",
            ]
            for k in keys:
                vals = [info.get(k) for info in infos if k in info]
                if len(vals) > 0:
                    self.logger.record(f"custom/{k}", float(np.mean(vals)))

        return True



def main():
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_envs = 8

    train_env = make_vec_env(
        env_id=DroneFormationEnv,
        n_envs=n_envs,
        env_kwargs=dict(
            gui=False,
            episode_len=500,
            use_wind=True,
            wind_std=0.2,
        ),
        monitor_dir="./logs/train_raw/",
    )

    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )

    eval_env_raw = make_vec_env(
        env_id=DroneFormationEnv,
        n_envs=1,
        env_kwargs=dict(
            gui=False,
            episode_len=500,
            use_wind=True,
            wind_std=0.2,
        ),
    )

    if os.path.exists("./models/vecnormalize.pkl"):
        eval_env = VecNormalize.load("./models/vecnormalize.pkl", eval_env_raw)
    else:
        eval_env = VecNormalize(
            eval_env_raw,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )
    eval_env.training = False
    eval_env.norm_reward = False

    model = SAC(
        policy=MlpPolicy,
        env=train_env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=512,
        gamma=0.99,
        tau=0.02,
        train_freq=4,
        gradient_steps=4,
        target_update_interval=1,
        device=device,
        tensorboard_log="./logs/tensorboard/",
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    os.makedirs("models/best", exist_ok=True)
    os.makedirs("logs/eval", exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=250_000,
        save_path="./models/",
        name_prefix="sac_formation",
        save_replay_buffer=True,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path="./models/best/",
        log_path="./logs/eval/",
        eval_freq=50_000,
        deterministic=True,
        render=False,
    )

    curriculum = CurriculumCallback(verbose=0)
    metrics = MetricsCallback(log_freq=1000, verbose=0)

    total_timesteps = 1_000_000

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback, curriculum, metrics],
        log_interval=10,
    )

    train_env.save("./models/vecnormalize.pkl")
    model.save("./models/sac_formation_final")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
