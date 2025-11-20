import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data


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
        self.g = np.array([0.0, 0.0, -9.81], dtype=np.float32)

        self.max_thrust = 12.0
        self.min_thrust = 0.0

        self.use_wind = use_wind
        self.wind_std = wind_std

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
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

        self.rng = np.random.default_rng()

        self.maze_map = [
            "#########",
            "#   #   #",
            "# # # # #",
            "# #   # #",
            "# ##### #",
            "#   #   #",
            "### # ###",
            "#       #",
            "#########",
        ]
        self.cell_size = 1.0
        self.rows = len(self.maze_map)
        self.cols = len(self.maze_map[0])

        if self.gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(self.dt, physicsClientId=self.client)

        self.plane_id = None
        self.drone1_id = None
        self.drone2_id = None
        self.wall_ids = []

        self.target = np.zeros(3, dtype=np.float32)

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

    def _cell_to_world(self, row, col, z=1.2):
        x = (col - (self.cols - 1) / 2.0) * self.cell_size
        y = (row - (self.rows - 1) / 2.0) * self.cell_size
        return np.array([x, y, z], dtype=np.float32)

    def _build_maze_world(self):
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(self.dt, physicsClientId=self.client)

        self.wall_ids = []
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)

        half_extents = [self.cell_size * 0.5, self.cell_size * 0.5, 0.75]
        col_wall = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.client
        )
        vis_wall = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[0.4, 0.4, 0.4, 1.0],
            physicsClientId=self.client,
        )

        for r in range(self.rows):
            for c in range(self.cols):
                if self.maze_map[r][c] == "#":
                    pos = self._cell_to_world(r, c, z=0.75)
                    wall_id = p.createMultiBody(
                        baseMass=0.0,
                        baseCollisionShapeIndex=col_wall,
                        baseVisualShapeIndex=vis_wall,
                        basePosition=pos.tolist(),
                        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                        physicsClientId=self.client,
                    )
                    self.wall_ids.append(wall_id)

        half_extents_drone = [0.15, 0.15, 0.05]
        col_drone = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half_extents_drone, physicsClientId=self.client
        )
        vis1 = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents_drone,
            rgbaColor=[0.1, 0.1, 0.9, 1.0],
            physicsClientId=self.client,
        )
        vis2 = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents_drone,
            rgbaColor=[0.1, 0.9, 0.1, 1.0],
            physicsClientId=self.client,
        )

        base_pos1 = self.p1.tolist()
        base_pos2 = self.p2.tolist()
        base_orn = p.getQuaternionFromEuler([0.0, 0.0, 0.0])

        self.drone1_id = p.createMultiBody(
            baseMass=self.mass,
            baseCollisionShapeIndex=col_drone,
            baseVisualShapeIndex=vis1,
            basePosition=base_pos1,
            baseOrientation=base_orn,
            physicsClientId=self.client,
        )
        self.drone2_id = p.createMultiBody(
            baseMass=self.mass,
            baseCollisionShapeIndex=col_drone,
            baseVisualShapeIndex=vis2,
            basePosition=base_pos2,
            baseOrientation=base_orn,
            physicsClientId=self.client,
        )

        p.resetBaseVelocity(
            self.drone1_id,
            linearVelocity=self.v1.tolist(),
            angularVelocity=self.omega1.tolist(),
            physicsClientId=self.client,
        )
        p.resetBaseVelocity(
            self.drone2_id,
            linearVelocity=self.v2.tolist(),
            angularVelocity=self.omega2.tolist(),
            physicsClientId=self.client,
        )

    def _motor_thrusts(self, a):
        u = 0.5 * (np.clip(a, -1.0, 1.0) + 1.0)
        return u * self.max_thrust

    def _get_state_from_bullet(self):
        pos1, orn1 = p.getBasePositionAndOrientation(self.drone1_id, physicsClientId=self.client)
        lin1, ang1 = p.getBaseVelocity(self.drone1_id, physicsClientId=self.client)
        pos2, orn2 = p.getBasePositionAndOrientation(self.drone2_id, physicsClientId=self.client)
        lin2, ang2 = p.getBaseVelocity(self.drone2_id, physicsClientId=self.client)

        eul1 = p.getEulerFromQuaternion(orn1)
        eul2 = p.getEulerFromQuaternion(orn2)

        self.p1 = np.array(pos1, dtype=np.float32)
        self.v1 = np.array(lin1, dtype=np.float32)
        self.ang1 = np.array(eul1, dtype=np.float32)
        self.omega1 = np.array(ang1, dtype=np.float32)

        self.p2 = np.array(pos2, dtype=np.float32)
        self.v2 = np.array(lin2, dtype=np.float32)
        self.ang2 = np.array(eul2, dtype=np.float32)
        self.omega2 = np.array(ang2, dtype=np.float32)

    def _apply_drone_forces(self, drone_id, motors):
        pos, orn = p.getBasePositionAndOrientation(drone_id, physicsClientId=self.client)
        R = np.array(p.getMatrixFromQuaternion(orn), dtype=np.float32).reshape(3, 3)
        thrust_body = np.array([0.0, 0.0, np.sum(motors)], dtype=np.float32)
        thrust_world = R @ thrust_body
        p.applyExternalForce(
            drone_id,
            -1,
            forceObj=thrust_world.tolist(),
            posObj=pos,
            flags=p.WORLD_FRAME,
            physicsClientId=self.client,
        )

        T1, T2, T3, T4 = motors
        l = self.arm
        k_yaw = 0.01
        tau_body = np.array(
            [
                l * (T2 - T4),
                l * (T3 - T1),
                k_yaw * (T1 - T2 + T3 - T4),
            ],
            dtype=np.float32,
        )
        tau_world = R @ tau_body
        p.applyExternalTorque(
            drone_id,
            -1,
            torqueObj=tau_world.tolist(),
            flags=p.WORLD_FRAME,
            physicsClientId=self.client,
        )

    def _get_obs(self):
        return np.concatenate(
            [
                self.p1,
                self.v1,
                self.ang1,
                self.omega1,
                self.p2,
                self.v2,
                self.ang2,
                self.omega2,
                self.target,
                self.offset,
            ]
        ).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0

        noise = self.init_noise

        start1 = (1, 1)
        start2 = (self.rows - 2, self.cols - 2)
        center = (self.rows // 2, self.cols // 2)

        self.p1 = self._cell_to_world(start1[0], start1[1], z=1.2) + self.rng.normal(
            0, noise, 3
        )
        self.v1 = self.rng.normal(0.0, 0.1, 3).astype(np.float32)
        self.ang1 = self.rng.normal(0.0, 0.05, 3).astype(np.float32)
        self.omega1 = self.rng.normal(0.0, 0.3, 3).astype(np.float32)

        self.p2 = self._cell_to_world(start2[0], start2[1], z=1.2) + self.rng.normal(
            0, noise, 3
        )
        self.v2 = self.rng.normal(0.0, 0.1, 3).astype(np.float32)
        self.ang2 = self.rng.normal(0.0, 0.05, 3).astype(np.float32)
        self.omega2 = self.rng.normal(0.0, 0.3, 3).astype(np.float32)

        self.target = self._cell_to_world(center[0], center[1], z=1.2)

        self._build_maze_world()
        self._get_state_from_bullet()

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        a = np.asarray(action, dtype=np.float32)
        m1 = self._motor_thrusts(a[:4])
        m2 = self._motor_thrusts(a[4:8])

        if self.use_wind:
            wind1 = self.rng.normal(0, self.wind_std, 3).astype(np.float32)
            wind2 = self.rng.normal(0, self.wind_std, 3).astype(np.float32)
        else:
            wind1 = np.zeros(3, dtype=np.float32)
            wind2 = np.zeros(3, dtype=np.float32)

        if np.linalg.norm(wind1) > 0:
            p.applyExternalForce(
                self.drone1_id,
                -1,
                forceObj=(wind1 * self.mass).tolist(),
                posObj=self.p1.tolist(),
                flags=p.WORLD_FRAME,
                physicsClientId=self.client,
            )
        if np.linalg.norm(wind2) > 0:
            p.applyExternalForce(
                self.drone2_id,
                -1,
                forceObj=(wind2 * self.mass).tolist(),
                posObj=self.p2.tolist(),
                flags=p.WORLD_FRAME,
                physicsClientId=self.client,
            )

        self._apply_drone_forces(self.drone1_id, m1)
        self._apply_drone_forces(self.drone2_id, m2)

        p.stepSimulation(physicsClientId=self.client)
        self._get_state_from_bullet()

        if self.target_moving:
            self.target += np.array([0.001, 0.002, 0.0005], dtype=np.float32)

        dist1 = np.linalg.norm(self.p1 - self.target)
        dist2 = np.linalg.norm(self.p2 - self.target)
        dist_target = 0.5 * (dist1 + dist2)

        form_error = np.linalg.norm((self.p2 - self.p1) - self.offset)
        tilt = (
            abs(self.ang1[0])
            + abs(self.ang1[1])
            + abs(self.ang2[0])
            + abs(self.ang2[1])
        )
        energy = float(np.sum(m1) + np.sum(m2))
        inter = np.linalg.norm(self.p2 - self.p1)

        reward = (
            -self.w_target * dist_target
            - self.w_form * form_error
            - self.w_tilt * tilt
            - self.w_energy * energy
        )

        contact1 = p.getContactPoints(self.drone1_id, physicsClientId=self.client)
        contact2 = p.getContactPoints(self.drone2_id, physicsClientId=self.client)
        collided = False
        for cp in contact1 + contact2:
            if cp[2] in self.wall_ids or cp[4] in self.wall_ids:
                collided = True
                break
        if collided:
            reward -= 10.0

        if inter < self.collision_dist:
            reward -= 5.0
        if self.p1[2] < 0.05 or self.p2[2] < 0.05:
            reward -= 5.0

        goal_reached = dist1 < 0.5 and dist2 < 0.5

        terminated = bool(goal_reached or collided)
        truncated = self.step_count >= self.episode_len

        info = {
            "dist_target": float(dist_target),
            "form_error": float(form_error),
            "tilt": float(tilt),
            "energy": float(energy),
            "inter_drone_dist": float(inter),
            "goal_reached": goal_reached,
            "collided": collided,
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        pass

    def close(self):
        if hasattr(self, "client") and self.client is not None:
            p.disconnect(self.client)
            self.client = None
