from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        step = self.num_timesteps


        if step == 100_000:
            self.training_env.env_method("set_wind_std", 0.20)
            self.training_env.env_method("set_init_noise", 0.10)
            self.training_env.env_method("set_collision_dist", 0.30)
            if self.verbose:
                print("\n[Curriculum] Stage 1 → Slight wind + lower noise + safer distance\n")


        if step == 250_000:
            self.training_env.env_method("set_wind_std", 0.30)
            self.training_env.env_method("set_init_noise", 0.15)
            self.training_env.env_method("set_collision_dist", 0.25)
            if self.verbose:
                print("\n[Curriculum] Stage 2 → Moderate wind + more noise\n")


        if step == 450_000:
            self.training_env.env_method("set_wind_std", 0.40)
            self.training_env.env_method("set_init_noise", 0.20)
            self.training_env.env_method("set_target_motion", True)
            if self.verbose:
                print("\n[Curriculum] Stage 3 → Moving target enabled\n")


        if step == 650_000:
            self.training_env.env_method("set_wind_std", 0.50)
            self.training_env.env_method("set_init_noise", 0.25)
            self.training_env.env_method("set_reward_weights", 1.3, 2.8, 0.07, 0.002)
            if self.verbose:
                print("\n[Curriculum] Stage 4 → Strong wind + harder reward shaping\n")


        if step == 850_000:
            self.training_env.env_method("set_wind_std", 0.60)
            self.training_env.env_method("set_init_noise", 0.30)
            self.training_env.env_method("set_collision_dist", 0.18)
            if self.verbose:
                print("\n[Curriculum] Stage 5 → Maximum difficulty\n")

        return True
