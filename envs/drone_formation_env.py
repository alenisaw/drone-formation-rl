import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DroneFormationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        gui: bool = False,
        episode_len: int = 600,
        use_wind: bool = True,
        wind_std: float = 0.3,
    ):
        super().__init__()

        self.dt = 0.02
        self.episode_len = episode_len
        self.gui = gui

        self.mass = 0.75
        self.arm = 0.18
        self.inertia = np.diag([0.02, 0.02, 0.04]).astype(np.float32)
        self.inv_inertia = np.linalg.inv(self.inertia)
        self.g = np.array([0., 0., -9.81], dtype=np.float32)

        self.max_thrust = 12.0
        self.min_thrust = 0.0
        self.max_lin_vel = 8.0
        self.max_ang_vel = 8.0

        self.use_wind = use_wind
        self.wind_std = wind_std

        self.target = np.array([0., 3., 1.2], dtype=np.float32)
        self.offset = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        self.collision_dist = 0.25
        self.w_target = 1.2
        self.w_form = 2.0
        self.w_tilt = 0.05
        self.w_energy = 0.001

        self.init_noise = 0.2
        self.target_moving = False

        self.obs_dim = 30
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        self.action_space = spaces.Box(low=-1., high=1., shape=(8,), dtype=np.float32)

        self.rng = np.random.default_rng()
        self.reset(seed=None)

    def set_wind_std(self, std):
        self.wind_std = float(std)

    def set_init_noise(self, level):
        self.init_noise = float(level)

    def set_target_motion(self, enabled):
        self.target_moving = bool(enabled)

    def set_collision_dist(self, d):
        self.collision_dist = float(d)

    def set_reward_weights(self, wt, wf, wi, we):
        self.w_target = float(wt)
        self.w_form = float(wf)
        self.w_tilt = float(wi)
        self.w_energy = float(we)

    def _euler_to_rot(self, ang):
        roll, pitch, yaw = ang
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        return np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]
        ], dtype=np.float32)

    def _motor_thrusts(self, a):
        u = 0.5 * (np.clip(a, -1, 1) + 1.0)
        return u * self.max_thrust

    def _drone_step(self, p, v, ang, omega, motors, wind):
        R = self._euler_to_rot(ang)
        thrust_body = np.array([0., 0., np.sum(motors)], dtype=np.float32)
        thrust_world = R @ thrust_body
        acc = thrust_world / self.mass + self.g + wind

        v = v + acc * self.dt
        speed = np.linalg.norm(v)
        if speed > self.max_lin_vel:
            v *= self.max_lin_vel / (speed + 1e-6)
        p = p + v * self.dt

        T1, T2, T3, T4 = motors
        l = self.arm
        k_yaw = 0.01
        tau = np.array([
            l * (T2 - T4),
            l * (T3 - T1),
            k_yaw * (T1 - T2 + T3 - T4)
        ], dtype=np.float32)

        omega = omega + (self.inv_inertia @ (tau - np.cross(omega, self.inertia @ omega))) * self.dt
        ang = ang + omega * self.dt

        w_norm = np.linalg.norm(omega)
        if w_norm > self.max_ang_vel:
            omega *= self.max_ang_vel / (w_norm + 1e-6)

        if p[2] < 0.0:
            p[2] = 0.0

        return p, v, ang, omega

    def _get_obs(self):
        return np.concatenate([
            self.p1, self.v1, self.ang1, self.omega1,
            self.p2, self.v2, self.ang2, self.omega2,
            self.target, self.offset
        ]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0

        noise = self.init_noise
        self.p1 = np.array([0., 0., 1.2]) + self.rng.normal(0, noise, 3)
        self.v1 = self.rng.normal(0., 0.1, 3).astype(np.float32)
        self.ang1 = self.rng.normal(0., 0.05, 3).astype(np.float32)
        self.omega1 = self.rng.normal(0., 0.3, 3).astype(np.float32)

        self.p2 = self.p1 + self.offset + self.rng.normal(0, noise * 0.5, 3)
        self.v2 = self.rng.normal(0., 0.1, 3).astype(np.float32)
        self.ang2 = self.rng.normal(0., 0.05, 3).astype(np.float32)
        self.omega2 = self.rng.normal(0., 0.3, 3).astype(np.float32)

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        a = np.asarray(action, dtype=np.float32)
        m1 = self._motor_thrusts(a[:4])
        m2 = self._motor_thrusts(a[4:8])

        wind1 = self.rng.normal(0, self.wind_std, 3).astype(np.float32) if self.use_wind else np.zeros(3)
        wind2 = self.rng.normal(0, self.wind_std, 3).astype(np.float32) if self.use_wind else np.zeros(3)

        self.p1, self.v1, self.ang1, self.omega1 = self._drone_step(self.p1, self.v1, self.ang1, self.omega1, m1, wind1)
        self.p2, self.v2, self.ang2, self.omega2 = self._drone_step(self.p2, self.v2, self.ang2, self.omega2, m2, wind2)

        if self.target_moving:
            self.target += np.array([0.001, 0.002, 0.0005], dtype=np.float32)

        dist_target = np.linalg.norm(self.p1 - self.target)
        form_error = np.linalg.norm((self.p2 - self.p1) - self.offset)
        tilt = abs(self.ang1[0]) + abs(self.ang1[1]) + abs(self.ang2[0]) + abs(self.ang2[1])
        energy = np.sum(m1) + np.sum(m2)
        inter = np.linalg.norm(self.p2 - self.p1)

        reward = (
            -self.w_target * dist_target
            - self.w_form * form_error
            - self.w_tilt * tilt
            - self.w_energy * energy
        )
        if inter < self.collision_dist:
            reward -= 5.0
        if self.p1[2] < 0.05 or self.p2[2] < 0.05:
            reward -= 5.0

        terminated = False
        truncated = self.step_count >= self.episode_len

        info = {
            "dist_target": float(dist_target),
            "form_error": float(form_error),
            "tilt": float(tilt),
            "energy": float(energy),
            "inter_drone_dist": float(inter),
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
