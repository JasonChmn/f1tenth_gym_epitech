"""RaceEnv — gymnasium wrapper for the Crash&Learn Grand Prix simulation."""

import math
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from env_simulation import (
    apply_action,
    close,
    get_obs,
    get_space_info,
    get_step_info,
    reset,
    simulation_step,
    _DECISION_FREQ_HZ
)

# ── Constants ──────────────────────────────────────────────────────────────────

_STEP_LIMIT = 8000  # decision steps; 8000 / 20Hz = 400s sim time per episode cap

# Toggle: set True to use pure pursuit controller, False for RL policy
USE_PURSUIT = True

# Pure pursuit defaults (overridden by set_controller)
_PP_SPEED = 2.0
_PP_LK    = 1.0   # lookahead distance (m)


class RaceEnv(gym.Env):
    """Single-agent racing environment wrapping env_simulation.

    Observation: [lidar / 15, velocity / 5, steering / 0.42]   shape=(102,)
    Action (raw): [-1, 1] denormalized to target_speed (-0.6 m/s reverse .. 5.0 m/s forward),
                   steering (-0.42 .. 0.42 rad)                     shape=(2,), bounds=[-1, 1]
    """

    metadata = {"render_modes": [None], "step_limit": _STEP_LIMIT}

    def __init__(self):
        super().__init__()

        self.space_info = get_space_info()
        
        # Extract observation bounds from space_info for per-component normalization
        obs_bounds = self.space_info["observations"]
        self._obs_lo = {
            "lidar": obs_bounds["lidar"]["bounds"][0],
            "velocity": obs_bounds["velocity"]["bounds"][0],
            "steering": obs_bounds["steering"]["bounds"][0],
        }
        self._obs_hi = {
            "lidar": obs_bounds["lidar"]["bounds"][1],
            "velocity": obs_bounds["velocity"]["bounds"][1],
            "steering": obs_bounds["steering"]["bounds"][1],
        }
        
        lidar_dim = self.space_info["observations"]["lidar"]["shape"][0]
        total_dim = lidar_dim + 1 + 1  # 102 = 100 + 1 + 1

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(total_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float64
        )

        self._speed_bounds = (
            self.space_info["actions"]["target_speed"]["bounds"]
        )  # (-0.6 reverse, 5.0 forward) m/s
        self._steering_bounds = (
            self.space_info["actions"]["steering"]["bounds"]
        )  # (-0.42 .. 0.42) rad

        self._step_count = 0
        self._last_progress = 0.0
        self._last_action = np.zeros(2, dtype=np.float32)
        self._controller = "pure_pursuit" if USE_PURSUIT else "rl"

    # ── Gymnasium interface ────────────────────────────────────────────────────

    def set_controller(self, mode: str) -> None:
        self._controller = mode

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        reset()
        raw = get_obs(0)
        self._last_progress = float(raw["progress"])
        self._step_count = 0
        self._last_action = np.zeros(2, dtype=np.float32)
        obs = self._get_processed_obs(raw)
        return obs, {}

    def _pure_pursuit_step(self):
        from env_simulation import _get_sim, _PARAMS as _EP
        sim_obj = _get_sim()
        # _waypoints / _arc / _total_arc are instance attrs on _Sim (set by __init__ → _load_waypoints)
        wp   = sim_obj._waypoints
        arc  = sim_obj._arc
        total_arc = sim_obj._total_arc
        # .agents lives on the internal Simulator (_sim), not on the wrapper
        agent = sim_obj._sim.agents[0]
        x, y, yaw = agent.state[0], agent.state[1], agent.state[4]
        dists = (wp[:, 0] - x) ** 2 + (wp[:, 1] - y) ** 2
        idx = int(np.argmin(dists))
        lk = self._lk
        s_lo, s_hi = _EP["s_min"], _EP["s_max"]
        spd = self._speed
        for i in range(1, len(wp)):
            wi = (idx + i) % len(wp)
            jx = (idx + i - 1) % len(wp)
            if math.sqrt((wp[wi,0]-wp[jx,0])**2 + (wp[wi,1]-wp[jx,1])**2) >= lk:
                tx, ty = wp[wi, 0], wp[wi, 1]
                angle = math.atan2(ty - y, tx - x) - yaw
                while angle > math.pi: angle -= 2 * math.pi
                while angle < -math.pi: angle += 2 * math.pi
                return spd, max(s_lo, min(s_hi, angle))
        tx, ty = wp[idx, 0], wp[idx, 1]
        angle = math.atan2(ty - y, tx - x) - yaw
        while angle > math.pi: angle -= 2 * math.pi
        while angle < -math.pi: angle += 2 * math.pi
        return spd, max(s_lo, min(s_hi, angle))

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        from env_simulation import _PARAMS as _EP
        
        if self._controller == "pure_pursuit":
            # Init pure pursuit params on first call
            if not hasattr(self, "_lk"):
                self._lk = float(_EP.get("lk", 1.0)) if isinstance(_EP.get("lk"), (int, float)) else 1.0
                self._speed = 2.0
            speed, steer = self._pure_pursuit_step()
            print("ctrl: PP")
            print("speed, steer : ",speed, " , ", steer)
        else:
            speed_raw = float(action[0])
            steering_raw = float(action[1])
            t_lo, t_hi = self._speed_bounds
            s_lo, s_hi = self._steering_bounds
            speed = t_lo + (speed_raw + 1.0) * (t_hi - t_lo) / 2.0
            steer = s_lo + (steering_raw + 1.0) * (s_hi - s_lo) / 2.0
            print("ctrl: RL")
        
        apply_action(0, speed, steer)
        #print(" Applied actions {}, {}".format(target_speed, steering))
        simulation_step()

        # Capture action info for smoothing reward regardless of controller
        if self._controller == "pure_pursuit":
            speed_raw_ = (speed - self._speed_bounds[0]) / ((self._speed_bounds[1] - self._speed_bounds[0]) / 2.0) - 1.0
            steering_raw_ = (steer - self._steering_bounds[0]) / ((self._steering_bounds[1] - self._steering_bounds[0]) / 2.0) - 1.0
        else:
            speed_raw_ = float(action[0])
            steering_raw_ = float(action[1])
        
        info = get_step_info()
        raw = get_obs(0)

        # ── Reward ──────────────────────────────────────────────────────────────
        dic_rewards = {
            "progress":         0,    # normalized to [0, 1]
            "smoothing":        0,    # normalized to [-1, 0]
            "collision_wall":   0,    # value {-1, 0}
            "stagnation":       0,    # value {-1, 0}
            "lap_bonus":        0,    # value {0, 1}
        }
        curr_progress = float(raw["progress"])
        delta = curr_progress - self._last_progress
        # Big enough to detect jump from ~0.99 (end) to ~0.01 (start)
        delta += 1 if delta < -0.5 else 0
        print("  progress: ",curr_progress)
        if delta > 0.:
            # 5m/s max at freq 25hz->0.04s ==> 5*0.04=0.2 meters
            t_lo, t_hi = self._speed_bounds # Suppose that t_hi > |t_lo|
            ratio = 1 / (t_hi / _DECISION_FREQ_HZ)
            dic_rewards["progress"] = delta * ratio
            self._last_progress = curr_progress # update only if going forward

        # Smoothing penalty — compare raw actions in [-1, 1] policy space (consistent with PPO output)
        action_delta = np.array([speed_raw_ - self._last_action[0], 
                                 steering_raw_ - self._last_action[1]])
        print("action_delta: ",action_delta)
        print("speed raw: ",speed_raw_)
        print("steering raw : ",steering_raw_)
        dic_rewards["smoothing"] = - float(np.sum(action_delta ** 2)/2)
        self._last_action = np.array([speed_raw_, steering_raw_])

        # Only agent id=0 here
        dic_rewards["collision_wall"] = -1 if info["collisions"]["wall"].get(0, False) else 0

        #  Stagnation / Done Condition 
        # Only agent id=0 here
        is_stagnant = info["stagnation"].get(0, False)
        dic_rewards["stagnation"] = -1 if is_stagnant else 0

        lap_finished = info["lap_complete"].get(0, False)
        dic_rewards["lap_bonus"] = 1 if lap_finished else 0

        dic_weights = {
            "progress": 100.0,
            "smoothing": 0.1,
            "collision_wall": 1.0,
            "stagnation": 1.0,
            "lap_bonus": 1000.0,
        }
        dic_rewards_weighted = {}
        reward = 0
        for key in dic_weights:
            v = dic_weights[key] * dic_rewards[key]
            dic_rewards_weighted[key] = v
            reward += v

        # ── Termination condition ────────────────────────────────────────
        self._step_count += 1
        terminated = lap_finished or is_stagnant
        truncated = bool(self._step_count >= _STEP_LIMIT)

        obs = self._get_processed_obs(raw)
        # Print the obs, reward, terminated, truncated, progression, rounded at second decimal
        if True:
            print("Obs: ",obs)
            print("Action: ",action)
            print(f"Reward: {reward:.2f}, Terminate: {terminated},", 
                f"Truncate: {truncated}, Progression: {curr_progress:.2f},",
                f"is_stagnant {is_stagnant}")
            print("Dic rew         : ",dic_rewards)
            print("Dic rew weighted: ",dic_rewards_weighted)
            #self.test_obs_scale()
            print("====")
        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        close()

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_processed_obs(self, raw: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Return normalized observation array of shape (102,) dtype=float32.
        
        Each component is normalized to [-1, 1] using its own lo/hi bounds from space_info:
            normalized = (val - lo) / (hi - lo) * 2 - 1
        """
        if raw is None:
            raw = get_obs(0)
        
        lidar_lo = self._obs_lo["lidar"]
        lidar_range = self._obs_hi["lidar"] - lidar_lo
        lidar = ((raw["lidar"] - lidar_lo) / lidar_range * 2.0 - 1.0).astype(np.float32)
        
        vel_lo = self._obs_lo["velocity"]
        vel_range = self._obs_hi["velocity"] - vel_lo
        velocity = np.array([(raw["velocity"] - vel_lo) / vel_range * 2.0 - 1.0], dtype=np.float32)
        
        steer_lo = self._obs_lo["steering"]
        steer_range = self._obs_hi["steering"] - steer_lo
        steering = np.array([(raw["steering"] - steer_lo) / steer_range * 2.0 - 1.0], dtype=np.float32)
        obs = np.concatenate([lidar, velocity, steering], dtype=np.float32)
        return np.clip(obs, -1.0, 1.0)
    
    def test_obs_scale(self):
        # Test by passing raw min/max values through _get_processed_obs()
        obs_bounds = self.space_info["observations"]
        
        # Raw values at minimum bounds
        raw_min = {
            "lidar": np.full(100, obs_bounds["lidar"]["bounds"][0]),
            "velocity": obs_bounds["velocity"]["bounds"][0],
            "steering": obs_bounds["steering"]["bounds"][0],
        }
        # Raw values at maximum bounds
        raw_max = {
            "lidar": np.full(100, obs_bounds["lidar"]["bounds"][1]),
            "velocity": obs_bounds["velocity"]["bounds"][1],
            "steering": obs_bounds["steering"]["bounds"][1],
        }
        
        obs_min = self._get_processed_obs(raw_min)
        obs_max = self._get_processed_obs(raw_max)
        
        print("obs_min (should be all -1.0):")
        print(f"  lidar[-5:] = {obs_min[:5]}...{obs_min[-5:]}")
        print(f"  velocity   = {obs_min[100]}")
        print(f"  steering   = {obs_min[101]}")
        
        print("obs_max (should be all +1.0):")
        print(f"  lidar[-5:] = {obs_max[:5]}...{obs_max[-5:]}")
        print(f"  velocity   = {obs_max[100]}")
        print(f"  steering   = {obs_max[101]}")
