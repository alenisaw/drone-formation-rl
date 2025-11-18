import os
import torch

torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"

from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from curriculum import CurriculumCallback
from envs.drone_formation_env import DroneFormationEnv


def main():
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_envs = 4

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

    eval_env = VecNormalize.load("./models/vecnormalize.pkl", eval_env_raw) if os.path.exists("./models/vecnormalize.pkl") else VecNormalize(
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
        save_freq=50000 // n_envs,
        save_path="./models/",
        name_prefix="sac_formation",
        save_replay_buffer=True,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path="./models/best/",
        log_path="./logs/eval/",
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    curriculum = CurriculumCallback(verbose=0)

    total_timesteps = 1_000_000

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback, curriculum],
    )

    train_env.save("./models/vecnormalize.pkl")
    model.save("./models/sac_formation_final")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
