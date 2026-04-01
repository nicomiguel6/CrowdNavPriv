"""
Maneuver test suite for tvbcbf.py

Tests the EvadeManeuver as a pure feedback controller.
Environment:
  - Robot starts at origin (0, 0), moving in +x, goal at (10, 0)
  - One motionless human blocking the lane at (5, 0)
  - Evade maneuver tracks a desired y-lane to pass the human, then returns
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from crowd_sim.envs.utils.state import FullState, ObservableState, JointState
from crowd_nav.policy.tvbcbf import EvadeManeuver


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

ROBOT_RADIUS = 0.3
HUMAN_RADIUS = 0.3
SAFETY_MARGIN = 0.1
MIN_CLEARANCE = ROBOT_RADIUS + HUMAN_RADIUS + SAFETY_MARGIN


def make_robot(px, py, vx, vy, gx=10.0, gy=0.0, v_pref=1.0):
    return FullState(
        px=px,
        py=py,
        vx=vx,
        vy=vy,
        radius=ROBOT_RADIUS,
        gx=gx,
        gy=gy,
        v_pref=v_pref,
        theta=np.arctan2(vy, vx),
    )


def step_robot(robot, vx, vy, dt):
    """Euler integrate robot position and return updated FullState."""
    speed = np.hypot(vx, vy)
    if speed > robot.v_pref:
        vx, vy = vx / speed * robot.v_pref, vy / speed * robot.v_pref

    return FullState(
        px=robot.px + vx * dt,
        py=robot.py + vy * dt,
        vx=vx,
        vy=vy,
        radius=robot.radius,
        gx=robot.gx,
        gy=robot.gy,
        v_pref=robot.v_pref,
        theta=np.arctan2(vy, vx),
    )


def dist_to_human(robot, human):
    return np.hypot(robot.px - human.px, robot.py - human.py)


def collision(robot, human):
    return dist_to_human(robot, human) < (robot.radius + human.radius)


# ---------------------------------------------------------------------------
# Evade maneuver test
# ---------------------------------------------------------------------------


def run_evade_test(
    evade_lane: float = 1.2,
    return_lane: float = 0.0,
    v_forward: float = 1.0,
    dt: float = 0.05,
    T_total: float = 20.0,
    goal_tol: float = 0.4,
    lane_tol: float = 0.1,
    human_clear_margin: float = 1.5,
):
    """
    Simulate the robot navigating from (0,0) to (10,0) around a human at (5,0).

    Phase 1 — Evade:
        The EvadeManeuver tracks `evade_lane` in y while maintaining forward
        speed.  Stays active until the robot has lateral clearance past the
        human.

    Phase 2 — Return:
        Once past the human, the EvadeManeuver is reused with `return_lane`
        (y = 0) as the target, guiding the robot back toward the x-axis while
        continuing forward.

    Returns
    -------
    traj : list of (px, py) tuples
    human : ObservableState
    meta : dict with collision flag and phase-change index
    """
    human = ObservableState(px=5.0, py=0.0, vx=0.0, vy=0.0, radius=HUMAN_RADIUS)
    robot = make_robot(px=0.0, py=0.0, vx=v_forward, vy=0.0)

    maneuver = EvadeManeuver()

    traj = [(robot.px, robot.py)]
    n_steps = int(T_total / dt)

    collided = False
    phase = 1  # 1 = evading, 2 = returning
    phase_change_idx = None

    for i in range(n_steps):
        # --- choose target lane based on phase ---
        if phase == 1:
            target_lane = evade_lane
            # switch to return phase once we have enough x-clearance past human
            if robot.px > human.px + human_clear_margin:
                phase = 2
                phase_change_idx = i
        else:
            target_lane = return_lane

        # --- evade maneuver gives [vx, vy_error] ---
        action = maneuver.compute_action(robot, target_lane)
        vx_cmd = action[0]  # preserves current forward speed
        vy_cmd = action[1]  # proportional y-error: (target - py)

        robot = step_robot(robot, vx_cmd, vy_cmd, dt)
        traj.append((robot.px, robot.py))

        if collision(robot, human):
            collided = True
            print(f"[!] Collision at step {i+1}  pos=({robot.px:.2f}, {robot.py:.2f})")
            break

        dx_goal = robot.gx - robot.px
        dy_goal = robot.gy - robot.py
        if np.hypot(dx_goal, dy_goal) < goal_tol:
            print(
                f"[+] Reached goal at step {i+1}  pos=({robot.px:.2f}, {robot.py:.2f})"
            )
            break

    return traj, human, {"collided": collided, "phase_change_idx": phase_change_idx}


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot_evade_trajectory(
    traj, human, meta, evade_lane: float = 1.2, title: str = "Evade Maneuver"
):
    traj = np.array(traj)
    phase_idx = meta.get("phase_change_idx")

    fig, ax = plt.subplots(figsize=(12, 5))

    # --- goal marker ---
    ax.plot(10.0, 0.0, "g*", markersize=18, label="Goal (10, 0)", zorder=5)

    # --- human ---
    human_circle = plt.Circle(
        (human.px, human.py),
        human.radius,
        color="red",
        alpha=0.4,
        label="Human (stationary)",
    )
    ax.add_patch(human_circle)
    ax.plot(human.px, human.py, "r+", markersize=10, zorder=5)

    # --- evade lane reference ---
    ax.axhline(
        evade_lane,
        color="orange",
        linestyle="--",
        linewidth=1.0,
        label=f"Evade lane y={evade_lane}",
    )
    ax.axhline(
        0.0,
        color="gray",
        linestyle=":",
        linewidth=0.8,
        alpha=0.6,
        label="Return lane y=0",
    )

    # --- trajectory ---
    if phase_idx is not None:
        ax.plot(
            traj[:phase_idx, 0],
            traj[:phase_idx, 1],
            "b-o",
            markersize=2,
            linewidth=1.5,
            label="Phase 1: evade",
        )
        ax.plot(
            traj[phase_idx:, 0],
            traj[phase_idx:, 1],
            "c-o",
            markersize=2,
            linewidth=1.5,
            label="Phase 2: return",
        )
    else:
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            "b-o",
            markersize=2,
            linewidth=1.5,
            label="Trajectory",
        )

    # --- start marker ---
    ax.plot(traj[0, 0], traj[0, 1], "bs", markersize=10, label="Start (0, 0)")

    # --- robot radius samples along trajectory ---
    sample_every = max(1, len(traj) // 12)
    for k in range(0, len(traj), sample_every):
        c = plt.Circle(
            (traj[k, 0], traj[k, 1]),
            ROBOT_RADIUS,
            color="blue",
            alpha=0.08,
        )
        ax.add_patch(c)

    ax.set_xlim(-0.5, 11)
    ax.set_ylim(-1.5, 2.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    collision_str = "YES" if meta["collided"] else "No"
    ax.text(
        0.99,
        0.03,
        f"Collision: {collision_str}",
        transform=ax.transAxes,
        ha="right",
        fontsize=9,
        color="red" if meta["collided"] else "green",
    )

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Evade Maneuver Test")
    print("  Robot: (0,0) -> goal (10,0) | Human at (5,0) stationary")
    print("=" * 60)

    traj, human, meta = run_evade_test(
        evade_lane=1.2,  # y-offset to dodge the human
        return_lane=0.0,  # y-target after passing the human
        v_forward=1.0,
        dt=0.05,
        T_total=20.0,
    )

    print(f"\nFinal position : ({traj[-1][0]:.3f}, {traj[-1][1]:.3f})")
    print(f"Steps simulated: {len(traj) - 1}")
    print(f"Collision      : {meta['collided']}")
    if meta["phase_change_idx"] is not None:
        print(f"Switched to return-lane at step {meta['phase_change_idx']}")

    fig = plot_evade_trajectory(traj, human, meta, evade_lane=1.2)
    out_path = os.path.join(os.path.dirname(__file__), "evade_test.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to: {out_path}")
    plt.show()
