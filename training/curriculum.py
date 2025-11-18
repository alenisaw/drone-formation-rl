from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        step = self.num_timesteps

        if step == 100_000:
            self.training_env.env_method("set_wind_std", 0.25)
            self.training_env.env_method("set_init_noise", 0.15)

        if step == 300_000:
            self.training_env.env_method("set_wind_std", 0.35)
            self.training_env.env_method("set_init_noise", 0.2)
            self.training_env.env_method("set_target_motion", True)

        if step == 600_000:
            self.training_env.env_method("set_wind_std", 0.45)
            self.training_env.env_method("set_collision_dist", 0.20)
            self.training_env.env_method("set_reward_weights", 1.2, 2.5, 0.07, 0.002)

        if step == 800_000:
            self.training_env.env_method("set_wind_std", 0.55)
            self.training_env.env_method("set_init_noise", 0.25)

        return True
