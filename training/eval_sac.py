import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from envs.drone_formation_env import DroneFormationEnv


def unpack_obs(raw):
    p1 = raw[0:3]
    v1 = raw[3:6]
    ang1 = raw[6:9]
    omega1 = raw[9:12]
    p2 = raw[12:15]
    v2 = raw[15:18]
    ang2 = raw[18:21]
    omega2 = raw[21:24]
    t = raw[24:27]
    off = raw[27:30]
    return p1, v1, ang1, omega1, p2, v2, ang2, omega2, t, off


def drone_mesh(px, py, pz, ang, size=0.3):
    roll, pitch, yaw = ang
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]
    ])
    arm1 = np.array([[size, 0, 0], [-size, 0, 0]]).T
    arm2 = np.array([[0, size, 0], [0, -size, 0]]).T
    arm1 = (R @ arm1).T + np.array([px, py, pz])
    arm2 = (R @ arm2).T + np.array([px, py, pz])
    return arm1, arm2


def load_training_metrics():
    if not os.path.exists("./logs/tensorboard/"):
        return None
    xs, ys_r, ys_c, ys_e = [], [], [], []
    for root, dirs, files in os.walk("./logs/tensorboard/"):
        for f in files:
            if f.endswith(".csv"):
                path = os.path.join(root, f)
                data = np.genfromtxt(path, delimiter=",", skip_header=1)
                if data.ndim == 1:
                    continue
                step = data[:, 1]
                val = data[:, 2]
                if "rollout/ep_rew_mean" in f:
                    xs.append(step); ys_r.append(val)
                if "train/critic_loss" in f:
                    xs.append(step); ys_c.append(val)
                if "train/ent_coef" in f:
                    xs.append(step); ys_e.append(val)
    return xs, ys_r, ys_c, ys_e


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env_raw = make_vec_env(
        DroneFormationEnv,
        n_envs=1,
        env_kwargs=dict(gui=False, episode_len=600, use_wind=True, wind_std=0.3),
    )

    vecnorm = VecNormalize.load("./models/vecnormalize.pkl", env_raw)
    vecnorm.training = False
    vecnorm.norm_reward = False

    model = SAC.load("./models/best/best_model.zip", env=vecnorm, device=device)

    obs = vecnorm.reset()

    p1s, p2s, ts = [], [], []
    v1s, v2s, a1s, a2s = [], [], [], []
    ang1s, ang2s = [], []
    rs, ds, fs, tls, ints = [], [], [], [], []

    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, dones, infos = vecnorm.step(action)
        done = bool(dones[0])
        info = infos[0]

        raw = vecnorm.get_original_obs()[0]
        p1, v1, ang1, _, p2, v2, ang2, _, target, _ = unpack_obs(raw)

        p1s.append(p1)
        p2s.append(p2)
        ts.append(target)
        v1s.append(v1)
        v2s.append(v2)
        ang1s.append(ang1)
        ang2s.append(ang2)

        thr1 = 0.5 * (action[0:4] + 1.0) * 12.0
        thr2 = 0.5 * (action[4:8] + 1.0) * 12.0
        a1s.append(thr1)
        a2s.append(thr2)

        rs.append(float(r[0]))
        ds.append(info["dist_target"])
        fs.append(info["form_error"])
        tls.append(info["tilt"])
        ints.append(info["inter_drone_dist"])

    p1s = np.array(p1s)
    p2s = np.array(p2s)
    ts = np.array(ts)
    v1s = np.array(v1s)
    v2s = np.array(v2s)
    ang1s = np.array(ang1s)
    ang2s = np.array(ang2s)
    a1s = np.array(a1s)
    a2s = np.array(a2s)
    rs = np.array(rs)
    ds = np.array(ds)
    fs = np.array(fs)
    tls = np.array(tls)
    ints = np.array(ints)

    os.makedirs("eval_outputs", exist_ok=True)

    report = {
        "total_reward": float(rs.sum()),
        "avg_reward": float(rs.mean()),
        "avg_dist_target": float(ds.mean()),
        "avg_form_error": float(fs.mean()),
        "avg_tilt": float(tls.mean()),
        "min_inter_drone_dist": float(ints.min()),
        "steps": len(rs),
    }
    with open("eval_outputs/report.json", "w") as f:
        json.dump(report, f, indent=4)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(p1s[:, 0], p1s[:, 1], p1s[:, 2])
    ax.plot(p2s[:, 0], p2s[:, 1], p2s[:, 2])
    ax.plot(ts[:, 0], ts[:, 1], ts[:, 2], "r--")
    plt.savefig("eval_outputs/traj.png", dpi=300)
    plt.close(fig)

    t = np.arange(len(rs))
    fig, axs = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    axs[0].plot(t, rs)
    axs[1].plot(t, ds); axs[1].plot(t, fs)
    axs[2].plot(t, np.linalg.norm(v1s, axis=1)); axs[2].plot(t, np.linalg.norm(v2s, axis=1))
    axs[3].plot(t, tls)
    plt.savefig("eval_outputs/time.png", dpi=300)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 2))
    plt.imshow(fs.reshape(1, -1), aspect="auto", cmap="magma")
    plt.colorbar()
    plt.savefig("eval_outputs/hm_form.png", dpi=300)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    line1x, = ax.plot([], [], [], "b-", linewidth=3)
    line1y, = ax.plot([], [], [], "b-", linewidth=3)
    line2x, = ax.plot([], [], [], "g-", linewidth=3)
    line2y, = ax.plot([], [], [], "g-", linewidth=3)

    xmin = min(p1s[:, 0].min(), p2s[:, 0].min()) - 1
    xmax = max(p1s[:, 0].max(), p2s[:, 0].max()) + 1
    ymin = min(p1s[:, 1].min(), p2s[:, 1].min()) - 1
    ymax = max(p1s[:, 1].max(), p2s[:, 1].max()) + 1
    zmin = 0
    zmax = max(p1s[:, 2].max(), p2s[:, 2].max()) + 1
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)

    def upd(i):
        arm1a, arm1b = drone_mesh(p1s[i, 0], p1s[i, 1], p1s[i, 2], ang1s[i])
        arm2a, arm2b = drone_mesh(p2s[i, 0], p2s[i, 1], p2s[i, 2], ang2s[i])

        line1x.set_data([arm1a[0, 0], arm1a[1, 0]], [arm1a[0, 1], arm1a[1, 1]])
        line1x.set_3d_properties([arm1a[0, 2], arm1a[1, 2]])
        line1y.set_data([arm1b[0, 0], arm1b[1, 0]], [arm1b[0, 1], arm1b[1, 1]])
        line1y.set_3d_properties([arm1b[0, 2], arm1b[1, 2]])

        line2x.set_data([arm2a[0, 0], arm2a[1, 0]], [arm2a[0, 1], arm2a[1, 1]])
        line2x.set_3d_properties([arm2a[0, 2], arm2a[1, 2]])
        line2y.set_data([arm2b[0, 0], arm2b[1, 0]], [arm2b[0, 1], arm2b[1, 1]])
        line2y.set_3d_properties([arm2b[0, 2], arm2b[1, 2]])

        return line1x, line1y, line2x, line2y

    ani = animation.FuncAnimation(fig, upd, frames=len(p1s), interval=50, blit=False)
    ani.save("eval_outputs/traj.gif", writer="pillow", fps=15)
    plt.close(fig)

    try:
        import plotly.graph_objs as go
        from plotly.offline import plot as pf

        tr1 = go.Scatter3d(x=p1s[:, 0], y=p1s[:, 1], z=p1s[:, 2], mode="lines")
        tr2 = go.Scatter3d(x=p2s[:, 0], y=p2s[:, 1], z=p2s[:, 2], mode="lines")
        tr3 = go.Scatter3d(x=ts[:, 0], y=ts[:, 1], z=ts[:, 2], mode="lines", line=dict(color="red"))
        fig = go.Figure(data=[tr1, tr2, tr3])
        fig.update_layout(title="3D Trajectory")
        pf(fig, filename="eval_outputs/traj_interactive.html", auto_open=False)
    except:
        pass


if __name__ == "__main__":
    main()
