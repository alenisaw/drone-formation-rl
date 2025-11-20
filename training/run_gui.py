import os
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.drone_formation_env import DroneFormationEnv

BASE_TRAIN_DIR = "training"
MODELS_DIR = os.path.join(BASE_TRAIN_DIR, "models")
VECNORM_PATH = os.path.join(MODELS_DIR, "vecnormalize.pkl")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best", "best_model.zip")


def make_env():
    return DroneFormationEnv(
        gui=True,
        episode_len=600,
        use_wind=True,
        wind_std=0.3,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    env = DummyVecEnv([make_env])
    vecnorm = VecNormalize.load(VECNORM_PATH, env)
    vecnorm.training = False
    vecnorm.norm_reward = False

    model = SAC.load(BEST_MODEL_PATH, env=vecnorm, device=device)

    obs = vecnorm.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vecnorm.step(action)
        done = bool(dones[0])

    env.close()


if __name__ == "__main__":
    main()
