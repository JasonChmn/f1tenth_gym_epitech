"""
demo.py — Headless smoke test: runs 200 decision steps and prints obs/info.
Validates the env_simulation API contract without any display.
"""
import numpy as np
from env_simulation import (
    get_space_info, reset, get_obs, apply_action, simulation_step, get_step_info, close,
)

def main():
    print("=== env_simulation demo ===\n")
    print("Space info:")
    space = get_space_info()
    print(f"  decision_freq_hz : {space['decision_freq_hz']}")
    print(f"  lidar shape      : {space['observations']['lidar']['shape']}")
    print(f"  action bounds    : speed {space['actions']['target_speed']['bounds']}, "
          f"steer {space['actions']['steering']['bounds']}")
    print()

    reset(num_cars=1)
    print("Reset with 1 car. Running 200 steps...")

    for step in range(200):
        speed   = 3.0
        steering = 0.05 * np.sin(step * 0.1)
        apply_action(0, speed, steering)
        simulation_step()
        info = get_step_info()
        obs  = get_obs(0)

        if step % 40 == 0:
            print(
                f"  step={info['step_count']:3d} | "
                f"vel={obs['velocity']:+.2f} m/s | "
                f"steer={obs['steering']:+.3f} rad | "
                f"progress={obs['progress']:.4f} | "
                f"lap={obs['lap_count']} | "
                f"wall={info['collisions']['wall'][0]} | "
                f"veh={info['collisions']['vehicle'][0]} | "
                f"friction={info['friction_current']:.3f}"
            )

    print("\nFinal obs keys  :", sorted(obs.keys()))
    print("lidar shape     :", obs['lidar'].shape)
    print("lidar dtype     :", obs['lidar'].dtype)
    print("velocity (scalar):", type(obs['velocity']).__name__)
    print("steering (scalar):", type(obs['steering']).__name__)
    print("progress (scalar):", type(obs['progress']).__name__)
    print("lap_count (int)  :", type(obs['lap_count']).__name__)
    print("rank (int)       :", type(obs['rank']).__name__)
    print("opponents keys  :", sorted(obs['opponents'].keys()))

    final_info = get_step_info()
    print("\nFinal info keys :", sorted(final_info.keys()))
    print("opponents_mask  :", final_info['opponents_mask'])
    print("agent_status    :", final_info['agent_status'])
    print("step_count      :", final_info['step_count'])

    assert obs['lidar'].shape == (100,), "LiDAR shape contract violated"
    assert isinstance(obs['velocity'], float), "velocity must be float scalar"
    assert isinstance(obs['steering'], float), "steering must be float scalar"
    assert isinstance(obs['progress'], float), "progress must be float scalar"
    assert isinstance(obs['lap_count'], int),  "lap_count must be int scalar"
    assert set(obs['opponents'].keys()) == {0, 1, 2, 3}, "opponents must have keys 0..3"
    assert "done" not in final_info, "info must NOT contain 'done'"
    assert "wall" in final_info["collisions"] and "vehicle" in final_info["collisions"]

    close()
    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
