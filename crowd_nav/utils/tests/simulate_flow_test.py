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
        maneuvers=[EvadeManeuver()], backup_mode="stop", T_M=0.5, delta=0.2
    )
    tbc = policy.tbcs[0]

    # print(traj)

    # plt.plot(traj[:, 0], traj[:, 1])
    plt.show()
