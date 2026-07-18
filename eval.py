import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import mujoco
from scipy.spatial.transform import Rotation as R

import time
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


CSV_JOINT_COLUMNS = [
    "left_hip_pitch_joint_dof",
    "left_hip_roll_joint_dof",
    "left_hip_yaw_joint_dof",
    "left_knee_joint_dof",
    "left_ankle_pitch_joint_dof",
    "left_ankle_roll_joint_dof",

    "right_hip_pitch_joint_dof",
    "right_hip_roll_joint_dof",
    "right_hip_yaw_joint_dof",
    "right_knee_joint_dof",
    "right_ankle_pitch_joint_dof",
    "right_ankle_roll_joint_dof",

    "waist_yaw_joint_dof",
    "waist_roll_joint_dof",
    "waist_pitch_joint_dof",

    "left_shoulder_pitch_joint_dof",
    "left_shoulder_roll_joint_dof",
    "left_shoulder_yaw_joint_dof",
    "left_elbow_joint_dof",
    "left_wrist_roll_joint_dof",
    "left_wrist_pitch_joint_dof",
    "left_wrist_yaw_joint_dof",

    "right_shoulder_pitch_joint_dof",
    "right_shoulder_roll_joint_dof",
    "right_shoulder_yaw_joint_dof",
    "right_elbow_joint_dof",
    "right_wrist_roll_joint_dof",
    "right_wrist_pitch_joint_dof",
    "right_wrist_yaw_joint_dof",
]


class G1MotionReference:
    def __init__(
        self,
        csv_path,
        model,
        mocap_fps=120.0,
        root_pos_scale=0.01,
        align_first_root_to_fallen=True,
    ):
        self.csv_path = csv_path
        self.model = model
        self.mocap_fps = mocap_fps
        self.dt = 1.0 / mocap_fps

        df = pd.read_csv(csv_path)

        missing = [c for c in CSV_JOINT_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing CSV columns: {missing}")

        self.n_frames = len(df)

        joint_deg = df[CSV_JOINT_COLUMNS].to_numpy(dtype=np.float64)
        self.qpos_joints = np.deg2rad(joint_deg)

        root_pos_raw = df[
            ["root_translateX", "root_translateY", "root_translateZ"]
        ].to_numpy(dtype=np.float64)

        root_pos = root_pos_raw * root_pos_scale

        root_euler_deg = df[
            ["root_rotateX", "root_rotateY", "root_rotateZ"]
        ].to_numpy(dtype=np.float64)

        root_rot = R.from_euler("xyz", root_euler_deg, degrees=True)

        fallen_qpos = self._get_key_qpos("fallen")
        fallen_root_pos = fallen_qpos[:3].copy()
        fallen_root_quat = fallen_qpos[3:7].copy()
        fallen_root_rot = self._quat_wxyz_to_rot(fallen_root_quat)

        if align_first_root_to_fallen:
            first_pos = root_pos[0].copy()

            root_pos[:, 0] = fallen_root_pos[0] + (root_pos[:, 0] - first_pos[0])
            root_pos[:, 1] = fallen_root_pos[1] + (root_pos[:, 1] - first_pos[1])
            root_pos[:, 2] = fallen_root_pos[2] + (root_pos[:, 2] - first_pos[2])

            first_rot = root_rot[0]
            offset_rot = fallen_root_rot * first_rot.inv()
            root_rot = offset_rot * root_rot

        quat_xyzw = root_rot.as_quat()
        quat_wxyz = np.zeros_like(quat_xyzw)
        quat_wxyz[:, 0] = quat_xyzw[:, 3]
        quat_wxyz[:, 1:] = quat_xyzw[:, :3]

        self.root_pos = root_pos
        self.root_quat = quat_wxyz

        self.qpos = np.zeros((self.n_frames, model.nq), dtype=np.float64)
        self.qpos[:, :3] = self.root_pos
        self.qpos[:, 3:7] = self.root_quat
        self.qpos[:, 7:] = self.qpos_joints

        self.qvel = np.zeros((self.n_frames, model.nv), dtype=np.float64)
        self.qvel[:, 6:] = self._finite_diff(self.qpos_joints, self.dt)

    def _get_key_qpos(self, key_name):
        key_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_KEY,
            key_name,
        )
        if key_id < 0:
            raise ValueError(f"Keyframe '{key_name}' not found in XML")
        return self.model.key_qpos[key_id].copy()

    @staticmethod
    def _quat_wxyz_to_rot(q):
        return R.from_quat([q[1], q[2], q[3], q[0]])

    @staticmethod
    def _finite_diff(x, dt):
        v = np.zeros_like(x)
        v[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
        v[0] = (x[1] - x[0]) / dt
        v[-1] = (x[-1] - x[-2]) / dt
        return v

    def frame_at_time(self, t):
        idx = int(round(t * self.mocap_fps))
        return int(np.clip(idx, 0, self.n_frames - 1))

    def get(self, idx):
        idx = int(np.clip(idx, 0, self.n_frames - 1))
        return self.qpos[idx], self.qvel[idx]

    def duration(self):
        return (self.n_frames - 1) / self.mocap_fps


class G1FallenStandEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        xml_path="g1.xml",
        mocap_csv_path="stand_up_lying_R_002__A475_new.csv",
        render_mode=None,
        mocap_fps=120.0,
        control_dt=1.0 / 60.0,
        action_scale=0.25,
        reset_noise_scale=0.005,
        random_start_frames=3,
        extra_stand_time=2.0,

        # Robustness / domain randomization parameters.
        # These can be increased during fine-tuning using VecEnv.set_attr(...).
        reset_joint_noise=0.005,
        reset_vel_noise=0.02,
        reset_root_xy_noise=0.0,
        reset_yaw_noise=0.0,
        friction_range=(0.6, 0.6),
        gravity_z_range=(-9.81, -9.81),
        push_prob=0.0,
        push_force_range=(0.0, 0.0),
        push_duration_range=(1, 1),
        push_start_phase_range=(0.15, 0.95),
    ):
        super().__init__()

        self.xml_path = xml_path
        self.mocap_csv_path = mocap_csv_path
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.ref = G1MotionReference(
            csv_path=mocap_csv_path,
            model=self.model,
            mocap_fps=mocap_fps,
            root_pos_scale=0.01,
            align_first_root_to_fallen=True,
        )

        self.control_dt = control_dt
        self.frame_skip = max(1, int(round(control_dt / self.model.opt.timestep)))
        self.actual_control_dt = self.frame_skip * self.model.opt.timestep

        self.action_scale = action_scale
        self.reset_noise_scale = reset_noise_scale  # kept for backward compatibility
        self.reset_joint_noise = reset_joint_noise
        self.reset_vel_noise = reset_vel_noise
        self.reset_root_xy_noise = reset_root_xy_noise
        self.reset_yaw_noise = reset_yaw_noise
        self.friction_range = friction_range
        self.gravity_z_range = gravity_z_range
        self.push_prob = push_prob
        self.push_force_range = push_force_range
        self.push_duration_range = push_duration_range
        self.push_start_phase_range = push_start_phase_range
        self.random_start_frames = random_start_frames

        self.extra_stand_steps = int(extra_stand_time / self.actual_control_dt)
        self.max_steps = int(np.ceil(self.ref.duration() / self.actual_control_dt)) + self.extra_stand_steps

        self.n_act = self.model.nu

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_act,),
            dtype=np.float32,
        )

        self.ctrl_low, self.ctrl_high = self._get_ctrl_limits()

        self.stand_qpos = self._get_key_qpos("stand")
        self.fallen_qpos = self._get_key_qpos("fallen")

        self.pelvis_body_id = self._get_body_id("pelvis")
        self.left_foot_body_id = self._get_body_id("left_ankle_roll_link")
        self.right_foot_body_id = self._get_body_id("right_ankle_roll_link")
        self.floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )

        self.step_count = 0
        self.current_push_steps_left = 0
        self.next_push_step = 10**9
        self.push_force = np.zeros(3, dtype=np.float64)
        self.start_ref_idx = 0
        self.last_action = np.zeros(self.n_act, dtype=np.float64)

        obs = self._get_obs()

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float32,
        )

        self.viewer = None

    def _get_key_qpos(self, key_name):
        key_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_KEY,
            key_name,
        )
        if key_id < 0:
            raise ValueError(f"Keyframe '{key_name}' not found")
        return self.model.key_qpos[key_id].copy()

    def _get_body_id(self, body_name):
        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )
        if body_id < 0:
            names = []
            for i in range(self.model.nbody):
                name = mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    i,
                )
                if name is not None:
                    names.append(name)

            raise ValueError(
                f"Body '{body_name}' not found. Available bodies:\n{names}"
            )

        return body_id

    def _get_ctrl_limits(self):
        if self.model.actuator_ctrllimited.any():
            low = self.model.actuator_ctrlrange[:, 0].copy()
            high = self.model.actuator_ctrlrange[:, 1].copy()
            return low, high

        low = np.full(self.n_act, -3.0, dtype=np.float64)
        high = np.full(self.n_act, 3.0, dtype=np.float64)

        for i in range(self.n_act):
            joint_id = self.model.actuator_trnid[i, 0]

            if joint_id >= 0 and self.model.jnt_limited[joint_id]:
                low[i] = self.model.jnt_range[joint_id, 0]
                high[i] = self.model.jnt_range[joint_id, 1]

        return low, high

    def _ref_index(self):
        t = self.step_count * self.actual_control_dt
        idx = self.start_ref_idx + self.ref.frame_at_time(t)
        return min(idx, self.ref.n_frames - 1)

    def _get_ref(self):
        idx = self._ref_index()
        return self.ref.get(idx)

    @staticmethod
    def _quat_dist(q1, q2):
        q1 = q1 / (np.linalg.norm(q1) + 1e-8)
        q2 = q2 / (np.linalg.norm(q2) + 1e-8)

        dot = abs(float(np.dot(q1, q2)))
        dot = np.clip(dot, -1.0, 1.0)

        return 2.0 * np.arccos(dot)

    def _upright_reward(self):
        root_mat = self.data.xmat[self.pelvis_body_id].reshape(3, 3)

        pelvis_z_axis = root_mat[:, 2]
        world_z_axis = np.array([0.0, 0.0, 1.0])

        upright = float(np.dot(pelvis_z_axis, world_z_axis))

        return float(np.clip((upright + 1.0) / 2.0, 0.0, 1.0))

    def _foot_slip_penalty(self):
        penalty = 0.0

        for body_id in [self.left_foot_body_id, self.right_foot_body_id]:
            foot_z = self.data.xpos[body_id, 2]

            # cvel layout: [angular_velocity, linear_velocity]
            foot_linear_vel = self.data.cvel[body_id][3:6]
            foot_xy_vel = np.linalg.norm(foot_linear_vel[:2])

            if foot_z < 0.06:
                penalty += foot_xy_vel ** 2

        return float(penalty)


    def _yaw_quat_wxyz(self, yaw):
        half = 0.5 * yaw
        return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)

    def _quat_mul_wxyz(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dtype=np.float64)

    def _sample_domain_randomization(self):
        # Randomize floor/foot friction and gravity per episode.
        friction = float(self.np_random.uniform(*self.friction_range))
        if self.floor_geom_id >= 0:
            self.model.geom_friction[self.floor_geom_id, 0] = friction

        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if name is None or "floor" in name:
                continue
            # Foot contact geoms are unnamed in this XML, so also randomize small sphere foot geoms.
            if self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE:
                self.model.geom_friction[geom_id, 0] = friction

        self.model.opt.gravity[:] = np.array([
            0.0,
            0.0,
            float(self.np_random.uniform(*self.gravity_z_range)),
        ])

    def _schedule_push_for_episode(self):
        self.current_push_steps_left = 0
        self.next_push_step = 10**9
        self.push_force[:] = 0.0

        if float(self.np_random.random()) >= self.push_prob:
            return

        min_phase, max_phase = self.push_start_phase_range
        min_step = int(np.clip(min_phase, 0.0, 1.0) * self.max_steps)
        max_step = int(np.clip(max_phase, 0.0, 1.0) * self.max_steps)
        if max_step <= min_step:
            max_step = min_step + 1

        self.next_push_step = int(self.np_random.integers(min_step, max_step))

        mag = float(self.np_random.uniform(*self.push_force_range))
        angle = float(self.np_random.uniform(-np.pi, np.pi))
        self.push_force[:] = mag * np.array([np.cos(angle), np.sin(angle), 0.0])

    def _apply_external_push(self):
        self.data.xfrc_applied[:, :] = 0.0

        if self.step_count == self.next_push_step:
            lo, hi = self.push_duration_range
            self.current_push_steps_left = int(self.np_random.integers(lo, hi + 1))

        if self.current_push_steps_left > 0:
            self.data.xfrc_applied[self.pelvis_body_id, :3] = self.push_force
            self.current_push_steps_left -= 1

    def _get_obs(self):
        ref_qpos, ref_qvel = self._get_ref()

        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        phase = self._ref_index() / max(1, self.ref.n_frames - 1)
        phase = min(phase, 1.0)

        joint_err = qpos[7:] - ref_qpos[7:]
        joint_vel_err = qvel[6:] - ref_qvel[6:]

        stand_joint_err = qpos[7:] - self.stand_qpos[7:]

        obs = np.concatenate(
            [
                qpos[2:],
                qvel,

                ref_qpos[2:],
                ref_qvel,

                joint_err,
                joint_vel_err,
                stand_joint_err,

                np.array([phase], dtype=np.float64),
                self.last_action,
            ]
        )

        return obs.astype(np.float32)

    def _compute_reward(self, action):
        ref_qpos, ref_qvel = self._get_ref()

        qpos = self.data.qpos
        qvel = self.data.qvel

        phase = self._ref_index() / max(1, self.ref.n_frames - 1)
        phase = min(phase, 1.0)

        joint_err = qpos[7:] - ref_qpos[7:]
        joint_vel_err = qvel[6:] - ref_qvel[6:]

        root_xy_err = qpos[:2] - ref_qpos[:2]
        root_z_err = qpos[2] - ref_qpos[2]
        root_quat_err = self._quat_dist(qpos[3:7], ref_qpos[3:7])

        pose_reward = np.exp(-4.0 * np.mean(joint_err ** 2))
        vel_reward = np.exp(-0.03 * np.mean(joint_vel_err ** 2))
        root_xy_reward = np.exp(-2.0 * np.mean(root_xy_err ** 2))
        height_reward = np.exp(-10.0 * root_z_err ** 2)
        root_ori_reward = np.exp(-1.0 * root_quat_err ** 2)

        upright_reward = self._upright_reward()

        stand_joint_err = qpos[7:] - self.stand_qpos[7:]
        stand_pose_reward = np.exp(-5.0 * np.mean(stand_joint_err ** 2))
        stand_height_reward = np.exp(-20.0 * (qpos[2] - self.stand_qpos[2]) ** 2)

        standing_weight = np.clip((phase - 0.65) / 0.35, 0.0, 1.0)
        tracking_weight = 1.0 - 0.5 * standing_weight

        foot_slip_penalty = self._foot_slip_penalty()
        ctrl_penalty = np.mean(action ** 2)
        smooth_penalty = np.mean((action - self.last_action) ** 2)

        tracking_reward = (
            2.0 * pose_reward
            + 0.7 * vel_reward
            + 0.2 * root_xy_reward
            + 1.0 * height_reward
            + 0.5 * root_ori_reward
        )

        final_stand_reward = (
            2.0 * stand_pose_reward
            + 2.0 * upright_reward
            + 1.5 * stand_height_reward
        )

        reward = (
            tracking_weight * tracking_reward
            + standing_weight * final_stand_reward
            + 0.2
            - 0.15 * foot_slip_penalty
            - 0.002 * ctrl_penalty
            - 0.005 * smooth_penalty
        )

        info = {
            "phase": float(phase),
            "pose_reward": float(pose_reward),
            "vel_reward": float(vel_reward),
            "height_reward": float(height_reward),
            "upright_reward": float(upright_reward),
            "stand_pose_reward": float(stand_pose_reward),
            "stand_height_reward": float(stand_height_reward),
            "foot_slip_penalty": float(foot_slip_penalty),
            "height": float(qpos[2]),
            "target_height": float(ref_qpos[2]),
            "standing_weight": float(standing_weight),
        }

        return float(reward), info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.last_action[:] = 0.0

        if self.random_start_frames > 0:
            self.start_ref_idx = int(
                self.np_random.integers(0, self.random_start_frames + 1)
            )
        else:
            self.start_ref_idx = 0

        self._sample_domain_randomization()
        self._schedule_push_for_episode()

        qpos, qvel = self.ref.get(self.start_ref_idx)

        qpos = qpos.copy()
        qvel = qvel.copy()

        qpos[0:2] += self.np_random.normal(
            0.0, self.reset_root_xy_noise, size=2
        )

        if self.reset_yaw_noise > 0.0:
            yaw = float(self.np_random.normal(0.0, self.reset_yaw_noise))
            qpos[3:7] = self._quat_mul_wxyz(self._yaw_quat_wxyz(yaw), qpos[3:7])
            qpos[3:7] /= np.linalg.norm(qpos[3:7]) + 1e-8

        qpos[7:] += self.np_random.normal(
            0.0,
            self.reset_joint_noise,
            size=qpos[7:].shape,
        )

        qvel[:6] += self.np_random.normal(
            0.0,
            self.reset_vel_noise,
            size=qvel[:6].shape,
        )

        qvel[6:] += self.np_random.normal(
            0.0,
            self.reset_vel_noise,
            size=qvel[6:].shape,
        )

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel

        self.data.ctrl[:] = np.clip(qpos[7:], self.ctrl_low, self.ctrl_high)

        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)

        ref_qpos, _ = self._get_ref()

        target_joint_qpos = ref_qpos[7:] + self.action_scale * action
        ctrl = np.clip(target_joint_qpos, self.ctrl_low, self.ctrl_high)

        self.data.ctrl[:] = ctrl

        self._apply_external_push()
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.data.xfrc_applied[:, :] = 0.0

        reward, info = self._compute_reward(action)

        self.step_count += 1

        height = self.data.qpos[2]

        bad_state = (
            not np.isfinite(self.data.qpos).all()
            or not np.isfinite(self.data.qvel).all()
            or height < -0.10
            or height > 1.80
        )

        terminated = bool(bad_state)
        truncated = bool(self.step_count >= self.max_steps)

        self.last_action = action.copy()

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return

        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

# -----------------------------------------------------------------------------
# Evaluation / visualization configuration
# -----------------------------------------------------------------------------

XML_PATH = "g1.xml"
CSV_PATH = "stand_up_lying_R_002__A475_new.csv"

MODEL_PATH = (
    "model/"
    "g1_fallen_stand_soft_ground.zip"
)
VECNORM_PATH = (
    "model/"
    "g1_fallen_stand_soft_ground_vecnormalize.pkl"
)


def make_env():
    """Create the environment used for deterministic visualization."""
    return G1FallenStandEnv(
        xml_path=XML_PATH,
        mocap_csv_path=CSV_PATH,
        render_mode=None,
        mocap_fps=120.0,
        control_dt=1.0 / 60.0,
        action_scale=0.25,

        # Clean evaluation reset.
        reset_joint_noise=0.0,
        reset_vel_noise=0.0,
        reset_root_xy_noise=0.0,
        reset_yaw_noise=0.0,

        # No external pushes during this visualization.
        push_prob=0.0,
        push_force_range=(0.0, 0.0),
        push_duration_range=(0, 0),
        push_start_phase_range=(0.20, 0.95),

        # Fixed rigid-ground properties.
        friction_range=(0.6, 0.6),
        gravity_z_range=(-9.81, -9.81),

        random_start_frames=0,
        extra_stand_time=2.0,
    )


def main():
    raw_env = make_env()
    vec_env = DummyVecEnv([lambda: raw_env])
    viewer = None

    try:
        vec_env = VecNormalize.load(VECNORM_PATH, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

        model = PPO.load(MODEL_PATH, env=vec_env, device="auto")
        obs = vec_env.reset()

        viewer = mujoco.viewer.launch_passive(raw_env.model, raw_env.data)

        dt = raw_env.actual_control_dt
        episode_reward = 0.0
        step_count = 0

        while viewer.is_running():
            loop_start = time.perf_counter()

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)

            episode_reward += float(reward[0])
            step_count += 1

            viewer.sync()

            if step_count % 30 == 0:
                env_info = info[0]
                print(
                    f"step={step_count:04d} "
                    f"reward={episode_reward:.2f} "
                    f"height={env_info.get('height', 0.0):.3f} "
                    f"target_height={env_info.get('target_height', 0.0):.3f} "
                    f"upright={env_info.get('upright_reward', 0.0):.3f} "
                    f"phase={env_info.get('phase', 0.0):.3f} "
                    f"standing_weight={env_info.get('standing_weight', 0.0):.3f}"
                )

            if done[0]:
                print(
                    "Episode ended. "
                    f"steps={step_count}, "
                    f"episode_reward={episode_reward:.2f}"
                )
                obs = vec_env.reset()
                episode_reward = 0.0
                step_count = 0

            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0.0, dt - elapsed))

    finally:
        if viewer is not None:
            viewer.close()
        vec_env.close()


if __name__ == "__main__":
    main()
