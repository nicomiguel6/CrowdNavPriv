"""
Test the simulator_flow function in tvbcbf.py

Tests the simulate_flow function in tvbcbf.py.
Environment:
  - Robot starts at origin (0, 0), moving in +x, goal at (10, 0)
  - One motionless human blocking the lane at (5, 0)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from crowd_sim.envs.utils.state import FullState, ObservableState, JointState
from crowd_nav.policy.tvbcbf import TVBCBF, EvadeManeuver

# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

ROBOT_RADIUS = 0.3
HUMAN_RADIUS = 0.3
SAFETY_MARGIN = 0.1
MIN_CLEARANCE = ROBOT_RADIUS + HUMAN_RADIUS + SAFETY_MARGIN


# ---------------------------------------------------------------------------
# Simulate flow test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Simulate Flow Test")
    print("  Robot: (0,0) -> goal (10,0) | Human at (5,0) stationary")
    print("=" * 60)

    robot = FullState(
        px=0.0,
        py=0.0,
        vx=1.0,
        vy=0.0,
        radius=ROBOT_RADIUS,
        gx=10.0,
        gy=0.0,
        v_pref=1.0,
        theta=0.0,
    )
    human = ObservableState(px=5.0, py=0.0, vx=0.0, vy=0.0, radius=HUMAN_RADIUS)
    state = JointState(robot, [human])

    policy = TVBCBF()
    policy.kinematics = "holonomic"
    policy.time_step = 0.25
    policy.build_default_tbcs(
        maneuvers=[EvadeManeuver()], backup_mode="stop", T_M=5.0, delta=1.0
    )
    tbc = policy.tbcs[0]

    # Simulate flow
    traj = policy.simulate_flow(robot, tbc, tau_0=0.0, T=10.0)

    # ---------------------------------------------------------------------------
    # Extract trajectory data
    # ---------------------------------------------------------------------------
    xs = np.array([s.px for s in traj])
    ys = np.array([s.py for s in traj])
    ts = np.arange(len(traj)) * policy.time_step

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot 1: XY trajectory ---
    ax1 = axes[0]
    ax1.plot(xs, ys, "b-o", markersize=3, linewidth=1.5, label="Robot path")
    ax1.plot(xs[0], ys[0], "gs", markersize=8, label="Start")
    ax1.plot(xs[-1], ys[-1], "r*", markersize=12, label="End")
    ax1.plot(robot.gx, robot.gy, "g^", markersize=10, label="Goal")
    ax1.add_patch(
        plt.Circle(
            (human.px, human.py),
            human.radius,
            color="orange",
            alpha=0.5,
            label="Human",
        )
    )
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title("XY Trajectory")
    ax1.legend()
    # ax1.set_aspect("equal")
    ax1.grid(True)

    # --- Plot 2: x and y vs time ---
    ax2 = axes[1]
    ax2.plot(ts, xs, "b-", linewidth=1.5, label="x position")
    ax2.plot(ts, ys, "r--", linewidth=1.5, label="y position")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Position (m)")
    ax2.set_title("Position vs Time")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("simulate_flow_trajectory.png", dpi=150)
    plt.show()
    print("Plots saved to simulate_flow_trajectory.png")
