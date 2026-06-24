#!/usr/bin/env python3
"""
trace_wall.py — Instrument the wall-collision smoke test to prove tunneling is a
POSITION phenomenon, not a velocity one.

Workflow:
  1. reset(1) → loop 300 decision steps with apply_action(0, 5.0, 0.0).
  2. Each step AFTER simulation_step() record CSV columns.
  3. Print displacement-per-step, first non-collision step, and penetration evidence.
  4. Plot trajectory + dist_to_obstacle vs step.

DO NOT modify env_simulation.py.
"""

import csv
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")          # headless
import matplotlib.pyplot as plt   # noqa: E402

# ------------------------------------------------------------------
# 1. Import the simulation module (relative to project root)
# ------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)          # env_simulation.py lives at project root
sys.path.insert(1, os.path.join(PROJECT_ROOT, "gym"))   # f110_gym package

import env_simulation as env   # noqa: E402
from f110_gym.envs.laser_models import distance_transform   # noqa: E402


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
NUM_STEPS = 300
CSV_PATH = os.path.join(PROJECT_ROOT, "recordings", "trace_wall.csv")
PNG_PATH = os.path.join(PROJECT_ROOT, "recordings", "trace_wall.png")

os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)


def main():
    # --------------------------------------------------------------
    # 2. Initialise simulation
    # --------------------------------------------------------------
    env.set_map("example", num_cars=1)
    env.reset(num_cars=1)

    # Reach the shared _Sim singleton and grab the RaceCar agent
    sim = env._get_sim()          # _Sim instance
    engine = sim._sim             # f110_gym.envs.base_classes.Simulator
    car  = engine.agents[0]       # RaceCar (ego)

    # ScanSimulator2D attributes for distance_transform lookup
    scan_sim = car.scan_simulator   # ScanSimulator2D (class-level singleton on RaceCar)

    # --------------------------------------------------------------
    # 3. Record per-step data
    # --------------------------------------------------------------
    rows = []                       # list of dicts for CSV rows

    prev_x = car.state[0]           # track forward displacement delta
    prev_y = car.state[1]

    collision_burst_start = None    # step where wall_hit first became True
    collision_burst_end   = None    # step where wall_hit returns False after burst
    first_penetration_step = None   # step where dist_to_obstacle <= 0 (inside wall)

    for step in range(NUM_STEPS):
        # Apply max forward, zero steer  (action format: [steer, target_speed])
        env.apply_action(agent_id=0, target_speed=5.0, steering=0.0)

        # Advance simulation
        env.simulation_step()

        # Collect data AFTER the step
        x = car.state[0]
        y = car.state[1]
        theta = car.state[4]
        vel = car.state[3]            # velocity — will be 0.0 if collision flagged

        step_info = env.get_step_info()
        wall_hit   = bool(step_info["collisions"]["wall"][0])
        vehicle_hit = bool(step_info["collisions"]["vehicle"][0])
        progress   = float(step_info["progress_delta"][0])
        status     = int(step_info["agent_status"][0])

        # --- dist_to_obstacle via distance_transform ---
        # Car center in world frame → query ScanSim2D dt map
        cx = car.state[0] +  (car.params["length"] / 2.0) * np.cos(theta)   # front-center x
        cy = car.state[1] +  (car.params["length"] / 2.0) * np.sin(theta)   # front-center y

        dist_obs = distance_transform(
            cx, cy,
            scan_sim.orig_x, scan_sim.orig_y,
            scan_sim.orig_c, scan_sim.orig_s,
            scan_sim.map_height, scan_sim.map_width,
            scan_sim.map_resolution, scan_sim.dt
        )

        # Forward displacement along heading (delta in x projected onto heading)
        dx = x - prev_x
        dy = y - prev_y
        delta_along_heading = dx * np.cos(theta) + dy * np.sin(np.pi/2 + theta - np.cos(theta))
        # Simpler: Euclidean forward displacement
        delta_forward = np.sqrt(dx*dx + dy*dy) * np.sign(dx * np.cos(theta) + dy * np.sin(theta))

        # Track sign of wall_hit burst
        if wall_hit and collision_burst_start is None:
            collision_burst_start = step
        elif not wall_hit and collision_burst_start is not None and collision_burst_end is None:
            collision_burst_end = step

        if dist_obs <= 0.0 and first_penetration_step is None:
            first_penetration_step = step

        rows.append({
            "step":             step,
            "x":                x,
            "y":                y,
            "velocity":         vel,
            "wall_hit":         int(wall_hit),
            "vehicle_hit":      int(vehicle_hit),
            "progress":         progress,
            "status":           status,
            "dist_to_obstacle": dist_obs,
            "delta_forward_m":  delta_forward,
        })

        prev_x = x
        prev_y = y

    # --------------------------------------------------------------
    # 4. Write CSV
    # --------------------------------------------------------------
    fieldnames = list(rows[0].keys())
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV written to: {CSV_PATH}")

    # --------------------------------------------------------------
    # 5. Summary statistics
    # --------------------------------------------------------------
    # Extract columns from rows (list of dicts) into numpy arrays
    steps       = np.array([r["step"]        for r in rows])
    vel_arr     = np.array([r["velocity"]    for r in rows])
    wall_arr    = np.array([r["wall_hit"]    for r in rows])
    dist_arr    = np.array([r["dist_to_obstacle"] for r in rows])
    delta_arr   = np.array([r["delta_forward_m"]   for r in rows])
    x_arr       = np.array([r["x"]           for r in rows])
    y_arr       = np.array([r["y"]           for r in rows])

    # Displacement during the collision burst (mm per decision step)
    if collision_burst_start is not None and collision_burst_end is not None:
        burst_mask = (steps >= collision_burst_start) & (steps <= collision_burst_end)
        total_displacement_mm = np.sum(np.abs(delta_arr[burst_mask])) * 1000.0
        avg_per_step_mm = np.mean(np.abs(delta_arr[burst_mask])) * 1000.0
    elif collision_burst_start is not None:
        # Burst extends to end of recording
        burst_mask = steps >= collision_burst_start
        total_displacement_mm = np.sum(np.abs(delta_arr[burst_mask])) * 1000.0
        avg_per_step_mm = np.mean(np.abs(delta_arr[burst_mask])) * 1000.0
    else:
        total_displacement_mm = 0.0
        avg_per_step_mm = 0.0

    # --- Print 5-line summary ---
    print("\n===== WALL-COLLISION TUNNELING TRACE SUMMARY =====")
    print(f"1. Collision burst starts at step {collision_burst_start}, ends at step {collision_burst_end}.")
    print(f"2. Forward displacement during burst: {total_displacement_mm:.4f} mm (avg {avg_per_step_mm:.4f} mm/step).")
    if first_penetration_step is not None:
        print(f"3. dist_to_obstacle <= 0 (inside wall) first at step {first_penetration_step}. "
              f"Recovered back to >0 at step {first_penetration_step + 1}: "
              f"dist={dist_arr[first_penetration_step]:.6f} → {dist_arr[min(first_penetration_step+1, len(dist_arr)-1)]:.6f}.")
    else:
        print(f"3. dist_to_obstacle never went <= 0 during recording (min = {dist_arr.min():.6f} m).")

    monotonic_creeper = False
    if collision_burst_start is not None and collision_burst_end is not None:
        burst_deltas = delta_arr[collision_burst_start:collision_burst_end+1]
        monotonic_creeper = all(d > 0 for d in burst_deltas)
    print(f"4. {'MONOTONIC forward creep' if monotonic_creeper else 'NON-MONOTONIC / oscillatory'} displacement during burst "
          f"(position tunneling: {'YES' if monotonic_creeper else 'unclear'}).")

    # Check if dist goes negative then jumps back up
    jumped_back = False
    for i in range(len(dist_arr) - 1):
        if dist_arr[i] < 0.0 and dist_arr[i+1] > 0.5:
            jumped_back = True
            break
    print(f"5. dist_to_obstacle penetrated then jumped back up: {'YES — instant escape through wall' if jumped_back else 'no clear penetration signature'}")

    # Also note that velocity==0 at collision steps is trivial (set by check_ttc)
    vel_zero_at_collision = np.all(vel_arr[wall_arr == 1] == 0.0)
    print(f"\n   NOTE: velocity==0 at collision steps is guaranteed by `state[3:]=0` in check_ttc().")
    print(f"   Velocity data alone proves nothing about position tunneling.")

    # --------------------------------------------------------------
    # 6. Plot trajectory + dist_to_obstacle
    # --------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [1, 1]})

    # --- Trajectory (x,y) ---
    ax0 = axes[0]
    ax0.plot(x_arr, y_arr, "-o", markersize=2, label="Trajectory")
    if collision_burst_start is not None:
        burst_mask = steps == collision_burst_start
        ax0.scatter(x_arr[burst_mask], y_arr[burst_mask], c="red", s=80, zorder=5, label="Collision start")
    ax0.set_xlabel("x (m)")
    ax0.set_ylabel("y (m)")
    ax0.set_title("Trajectory — Wall Collision Trace")
    ax0.legend()
    ax0.set_aspect("equal")

    # --- dist_to_obstacle vs step ---
    ax1 = axes[1]
    ax1.plot(steps, dist_arr, "-b", linewidth=1, label="dist_to_obstacle")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.7)
    if collision_burst_start is not None:
        ax1.axvspan(collision_burst_start, NUM_STEPS - 1, alpha=0.15, color="red", label="Collision period")
    if first_penetration_step is not None:
        ax1.scatter([first_penetration_step], [dist_arr[first_penetration_step]], c="magenta", s=60, zorder=5, label="First penetration (<=0)")
    ax1.set_xlabel("Decision step")
    ax1.set_ylabel("dist_to_obstacle (m)")
    ax1.set_title("Distance to Nearest Obstacle at Car Front-Center")
    ax1.legend()

    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=150)
    print(f"\nPlot saved to: {PNG_PATH}")

    # Cleanup
    env.close()


if __name__ == "__main__":
    main()