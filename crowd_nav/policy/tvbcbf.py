"""
Time-Varying Backup Control Barrier Function (TVBCBF) Policy

Implements the safety filtering framework from:
    Singletary et al., "Safe Drone Flight with Time-Varying Backup Controllers,"
    IEEE/RSJ IROS 2022.

Key concepts:
    - BackupManeuver: a maneuver uM(x) executed for time TM before engaging backup.
    - TimeVaryingBackupController: the composite policy pi(x, tau) from eq. (9),
      sequencing maneuver -> transition -> backup controller.
    - TVBCBF: the top-level Policy that manages multiple TBCs, computes the
      time-offset tau_0 (Algorithm 1), switches maneuvers (Algorithm 2),
      and blends desired/backup actions via the regulation function (eq. 7-8).
"""

import abc
import logging
import numpy as np
from numpy.linalg import norm
from scipy.integrate import solve_ivp

from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY, ActionRot
from crowd_sim.envs.utils.state import FullState, ObservableState

from typing import Optional, List

# ---------------------------------------------------------------------------
# Backup Maneuver Definitions
# ---------------------------------------------------------------------------


class BackupManeuver(abc.ABC):
    """
    Abstract base for a maneuver uM(x) that is executed for duration T_M
    before the backup controller engages.

    Subclass this to define domain-specific maneuvers (carry-on, evade, etc.).
    The maneuver must be state-feedback only (not time-varying) so that the
    time-offset tau_0 can be freely reset without causing discontinuities
    (Remark 1 in the paper).
    """

    def __init__(self, name: str = "maneuver"):
        self.name = name

    @abc.abstractmethod
    def compute_action(self, robot_state, human_states, **kwargs):
        """
        Return the maneuver action as a numpy array [vx, vy] (holonomic)
        or [v, r] (unicycle).

        Parameters
        ----------
        robot_state : FullState
        human_states : list[ObservableState]
        **kwargs : extra context (e.g. desired action for carry-on)
        """

    def __repr__(self):
        return f"BackupManeuver({self.name})"


class StopManeuver(BackupManeuver):
    """Maneuver: zero velocity (equivalent to jumping straight to backup)."""

    def __init__(self):
        super().__init__(name="stop")

    def compute_action(self, robot_state, human_states, **kwargs):
        return np.array([0.0, 0.0])


class CarryOnManeuver(BackupManeuver):
    """
    Carry-on maneuver (eq. 30): propagate the current desired input forward.
    The held action is updated every time tau_0 is reset.
    """

    def __init__(self):
        super().__init__(name="carry_on")
        self.held_action = np.array([0.0, 0.0])

    def latch_desired_action(self, u_des):
        """Called when tau_0 resets to capture the current desired input."""
        self.held_action = np.array(u_des, dtype=float)

    def compute_action(self, robot_state, human_states, **kwargs):
        return self.held_action.copy()


class EvadeManeuver(BackupManeuver):
    """
    Evade maneuver (eq. 31): reposition away from the nearest obstacle
    before stopping.  Direction and speed are configurable.

    Simpler: track to desired x/y lane while holding desired speed forward
    """

    def __init__(self, evade_direction=None):
        super().__init__(name="evade")
        self.evade_direction = evade_direction

    # For now, we want a simpler method that does not depend on the human states
    def compute_action(self, robot_state, evade_position, **kwargs):

        # Desired position lane (since we are always moving in the x direction, we want this to be a jump in the y direction)
        desired_maneuver = np.array([robot_state.vx, 1.0])

        return desired_maneuver

    # def compute_action(self, robot_state, human_states, **kwargs):
    #     if self.evade_direction is not None:
    #         d = np.array(self.evade_direction, dtype=float)
    #         d = d / (norm(d) + 1e-8)
    #         return self.evade_speed * d

    #     if len(human_states) == 0:
    #         return np.array([0.0, 0.0])

    #     nearest = min(
    #         human_states,
    #         key=lambda h: norm([robot_state.px - h.px, robot_state.py - h.py]),
    #     )
    #     away = np.array([robot_state.px - nearest.px, robot_state.py - nearest.py])
    #     d = norm(away)
    #     if d < 1e-6:
    #         perp = np.array([1.0, 0.0])
    #     else:
    #         unit = away / d
    #         perp = np.array([-unit[1], unit[0]])
    #     return self.evade_speed * perp


# ---------------------------------------------------------------------------
# Backup Controller (uB)
# ---------------------------------------------------------------------------


class BackupController:
    """
    The terminal backup controller uB(x) that drives the system into the
    backup set S_B.  For crowd navigation this is a simple "decelerate to
    stop" or "retreat from nearest obstacle" controller.
    """

    def __init__(self, mode="stop", gain=1.0):
        self.mode = mode
        self.gain = gain

    def compute_action(self, robot_state, human_states):
        if self.mode == "stop":
            return np.array([0.0, 0.0])

        # if self.mode == "retreat" and len(human_states) > 0:
        #     nearest = min(
        #         human_states,
        #         key=lambda h: norm([robot_state.px - h.px, robot_state.py - h.py]),
        #     )
        #     away = np.array([robot_state.px - nearest.px, robot_state.py - nearest.py])
        #     d = norm(away)
        #     if d < 1e-6:
        #         return np.array([0.0, 0.0])
        #     direction = away / d
        #     v_retreat = min(robot_state.v_pref, self.gain / d)
        #     return v_retreat * direction

        return np.array([0.0, 0.0])


# ---------------------------------------------------------------------------
# Time-Varying Backup Controller  (eq. 9)
# ---------------------------------------------------------------------------


class TimeVaryingBackupController:
    """
    Composite policy pi(x, tau) that sequences:
        tau <= T_M          :  maneuver uM(x)
        T_M < tau <= T_M+d  :  smooth transition uM->B(x, tau)
        tau > T_M+d         :  backup controller uB(x)

    Parameters
    ----------
    maneuver : BackupManeuver
    backup : BackupController
    T_M : float   — maneuver duration
    delta : float  — transition duration for smooth switching
    """

    def __init__(self, maneuver, backup, T_M=0.5, delta=0.2):
        self.maneuver = maneuver
        self.backup = backup
        self.T_M = T_M
        self.delta = delta

    @property
    def name(self):
        return self.maneuver.name

    def evaluate(self, tau, robot_state, human_states=None, **kwargs):
        """
        Evaluate pi(x, tau) at the given internal clock value tau.

        Returns action as np.ndarray of shape (2,).
        """
        if human_states is None:
            human_states = []
        u_m = self.maneuver.compute_action(robot_state, human_states, **kwargs)
        u_b = self.backup.compute_action(robot_state, human_states)

        if tau <= self.T_M:
            return u_m
        elif tau <= self.T_M + self.delta:
            alpha = (tau - self.T_M) / self.delta
            return (1.0 - alpha) * u_m + alpha * u_b
        else:
            return u_b

    def get_backup_action(self, robot_state, human_states):
        """Direct access to the backup controller action uB(x)."""
        return self.backup.compute_action(robot_state, human_states)

    def __repr__(self):
        return (
            f"TBC(maneuver={self.maneuver.name}, "
            f"T_M={self.T_M}, delta={self.delta})"
        )


# ---------------------------------------------------------------------------
# Safety Primitives (safe set S, backup set S_B)
# ---------------------------------------------------------------------------


class SafetyConstraints:
    """
    Evaluates h(x) >= 0  for safe set S  and  hB(x) >= 0 for backup set S_B.
    Provides the building blocks needed by the implicit CBF computation.
    """

    def __init__(self, safety_radius=0.2, backup_speed_threshold=0.1):
        self.safety_radius = safety_radius
        self.backup_speed_threshold = backup_speed_threshold

    def h_safe(self, robot_state, human_states):
        """
        Per-agent safety constraint  h_{ij}(x) = d_ij - d_safe.
        Returns a list of scalar values, one per human.
        """
        values = []
        for hs in human_states:
            d = norm([robot_state.px - hs.px, robot_state.py - hs.py])
            d_safe = robot_state.radius + hs.radius + self.safety_radius
            values.append(d - d_safe)
        return values

    def h_backup(self, robot_state):
        """
        Backup set constraint  hB(x) = threshold - ||v||.
        Satisfied when the agent is nearly stopped.
        """
        speed = norm([robot_state.vx, robot_state.vy])
        return self.backup_speed_threshold - speed

    def in_safe_set(self, robot_state, human_states):
        vals = self.h_safe(robot_state, human_states)
        return all(v >= 0 for v in vals)

    def in_backup_set(self, robot_state):
        return self.h_backup(robot_state) >= 0


# ---------------------------------------------------------------------------
# TVBCBF Policy
# ---------------------------------------------------------------------------


class TVBCBF(Policy):
    """
    Time-Varying Backup CBF safety filter.

    This policy wraps a *nominal* (desired) policy and filters its actions
    through the TBC framework.  Multiple TimeVaryingBackupControllers can
    be registered; the policy manages switching between them (Algorithm 2)
    and computes the time-offset tau_0 (Algorithm 1).

    Usage
    -----
    >>> tvbcbf = TVBCBF()
    >>> tvbcbf.configure(config)
    >>> tvbcbf.set_nominal_policy(some_orca_or_rl_policy)
    >>> tvbcbf.register_tbc(tbc_carry_on)
    >>> tvbcbf.register_tbc(tbc_evade)
    >>> action = tvbcbf.predict(joint_state)
    """

    def __init__(self):
        super().__init__()
        self.name = "TVBCBF"
        self.trainable = False
        self.multiagent_training = None
        self.kinematics = "holonomic"

        # integration options
        self.int_options = {"rtol": 1e-6, "atol": 1e-6}

        # Nominal (desired) policy whose actions we filter
        self.nominal_policy = None

        # Registered time-varying backup controllers {index: TBC}
        self.tbcs = []
        self.active_tbc_index = 0

        # Time-offset state  (Algorithm 1 / 2)
        self.tau_0 = 0.0
        self.system_time = 0.0

        # Regulation function parameter (eq. 8)
        self.beta = 3.0

        # Time horizons
        self.T = 2.0  # total backup horizon
        self.dt = 0.25  # discrete time increment (Delta)

        # Safety primitives
        self.safety = SafetyConstraints()

        # 2-D single-integrator matrices  (state = [px, py], u = [vx, vy])
        #   ẋ = A x + B u  →  ṗ = v  (velocity is the direct control input)
        self.A = np.zeros((2, 2), dtype=float)
        self.B = np.eye(2, dtype=float)

    # ------------------------------------------------------------------
    # Configuration & setup
    # ------------------------------------------------------------------

    def configure(self, config):
        if config.has_section("tvbcbf"):
            self.beta = config.getfloat("tvbcbf", "beta")
            self.T = config.getfloat("tvbcbf", "backup_horizon")
            self.dt = config.getfloat("tvbcbf", "dt")
            self.safety.safety_radius = config.getfloat("tvbcbf", "safety_radius")
            self.safety.backup_speed_threshold = config.getfloat(
                "tvbcbf", "backup_speed_threshold"
            )

        if config.has_section("action_space"):
            self.kinematics = config.get("action_space", "kinematics")

    def set_nominal_policy(self, policy):
        """Attach the desired / nominal policy to be safety-filtered."""
        self.nominal_policy = policy
        if policy is not None:
            self.kinematics = policy.kinematics

    def register_tbc(self, tbc):
        """Add a TimeVaryingBackupController to the available set."""
        self.tbcs.append(tbc)
        logging.info("TVBCBF: registered TBC %s (index %d)", tbc, len(self.tbcs) - 1)

    def build_default_tbcs(
        self,
        maneuvers: Optional[List[BackupManeuver]] = None,
        backup_mode: str = "stop",
        backup_gain: float = 1.0,
        T_M: float = 0.5,
        delta: float = 0.2,
    ):
        """Convenience: build and register the standard carry-on + evade pair."""
        backup = BackupController(mode=backup_mode, gain=backup_gain)
        if (
            maneuvers is None
        ):  # assume just stopping backup controller, defaults to basic backup CBF method
            maneuvers = [StopManeuver()]
        for maneuver in maneuvers:
            self.register_tbc(TimeVaryingBackupController(maneuver, backup, T_M, delta))

    # ------------------------------------------------------------------
    # Forwarding hooks to nominal policy
    # ------------------------------------------------------------------

    def set_phase(self, phase):
        self.phase = phase
        if self.nominal_policy is not None:
            self.nominal_policy.set_phase(phase)

    def set_device(self, device):
        self.device = device
        if self.nominal_policy is not None:
            self.nominal_policy.set_device(device)

    def set_env(self, env):
        self.env = env
        if self.nominal_policy is not None:
            self.nominal_policy.set_env(env)

    # ------------------------------------------------------------------
    # Core predict loop
    # ------------------------------------------------------------------

    def predict(self, state):
        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == "holonomic" else ActionRot(0, 0)

        if len(self.tbcs) == 0:
            raise RuntimeError(
                "TVBCBF: no backup controllers registered. "
                "Call register_tbc() or build_default_tbcs() first."
            )

        robot = state.self_state
        humans = state.human_states
        tbc = self.tbcs[self.active_tbc_index]

        # 1) Desired action from nominal policy
        u_des = self._get_desired_action(state)

        # 2) Update time-offset  (Algorithm 1)
        self.tau_0 = self.compute_time_offset(
            robot, humans, tbc, self.system_time, self.dt, self.tau_0
        )

        # If there is only one maneuver, we don't need to switch
        if len(self.tbcs) == 1:
            tbc = self.tbcs[0]
        else:
            # 3) Possibly switch maneuver  (Algorithm 2 — user implements body)
            self.active_tbc_index, self.tau_0 = self.switch_maneuver(
                robot,
                humans,
                self.tbcs,
                self.active_tbc_index,
                self.system_time,
                self.dt,
                self.tau_0,
            )
            tbc = self.tbcs[self.active_tbc_index]

        # 4) Compute implicit CBF value  h_I(x)
        h_I = self.compute_implicit_cbf(robot, humans, tbc, self.tau_0)

        # 5) Regulation function  (eq. 7-8)
        u_backup = tbc.get_backup_action(robot, humans)
        u_act = self.regulation_function(u_des, u_backup, h_I)

        # 6) Latch desired action for carry-on maneuver when tau_0 was reset
        if isinstance(tbc.maneuver, CarryOnManeuver):
            tbc.maneuver.latch_desired_action(u_des)

        self.system_time += self.dt

        if self.kinematics == "holonomic":
            return ActionXY(float(u_act[0]), float(u_act[1]))
        else:
            return ActionRot(float(u_act[0]), float(u_act[1]))

    # ------------------------------------------------------------------
    # Placeholder: Time-offset computation  (Algorithm 1)
    # ------------------------------------------------------------------

    def compute_time_offset(self, robot_state, human_states, tbc, t, dt, tau_0_prev):
        """
        Algorithm 1 — Online approximation of tau*_0.

        Determines whether the time-offset can be reset to -t (allowing
        the full maneuver) or must advance with the system clock.

        Parameters
        ----------
        robot_state  : FullState — current robot state x
        human_states : list[ObservableState] — observable humans
        tbc          : TimeVaryingBackupController — active TBC (pi)
        t            : float — current system time
        dt           : float — discrete time increment (Delta)
        tau_0_prev   : float — previous tau*_0 solution

        Returns
        -------
        tau_0 : float — updated time-offset

        Reference pseudocode (from paper):
            P  <- {Phi_pi(x, t+tau, -t) | 0 <= tau <= T}
            if (P subset S) and (Phi_pi(x, t+T, -t) in S_B):
                return -t
            else:
                return tau_0_prev + dt
        """
        # Reset candidate: tau_0 = -t gives the full maneuver window

        # 1) Propagate the system state forward using the TBC
        traj = self.simulate_flow(robot_state, tbc, tau_0=-t, T=self.T)
        if self._tbc_is_feasible(traj):
            return -t
        return tau_0_prev + dt

    # ------------------------------------------------------------------
    # Placeholder: Maneuver switching  (Algorithm 2)
    # ------------------------------------------------------------------

    def switch_maneuver(
        self, robot_state, human_states, tbcs, current_index, t, dt, tau_0_prev
    ):
        """
        Algorithm 2 — Maneuver switching.

        When the current TBC has exhausted its maneuver phase
        (tau_0 >= t - T_M - dt), try the next maneuver in round-robin
        order.  If the new maneuver allows a tau_0 reset, switch to it.

        Parameters
        ----------
        robot_state  : FullState
        human_states : list[ObservableState]
        tbcs         : list[TimeVaryingBackupController]
        current_index: int — index of the currently active TBC
        t            : float — system time
        dt           : float — discrete time increment
        tau_0_prev   : float — previous tau*_0

        Returns
        -------
        (new_index, tau_0) : tuple[int, float]

        Reference pseudocode (from paper):
            if tau_0_prev >= t - T_M - dt:
                j <- (j + 1) mod J
                P_j <- {Phi_{pi_j}(x, t+tau, -t) | 0 <= tau <= T}
                if (P_j subset S) and (Phi_{pi_j}(x, t+T, -t) in S_B):
                    return (j, -t)
                else:
                    return (j, tau_0_prev + dt)
            return (current_index, tau_0_prev)
        """
        tbc = tbcs[current_index]
        if tau_0_prev >= t - tbc.T_M - dt:
            j = (current_index + 1) % len(tbcs)
            if self._tbc_is_feasible(robot_state, human_states, tbcs[j], tau_0=-t):
                return j, -t
            return j, tau_0_prev + dt
        return current_index, tau_0_prev

    # ------------------------------------------------------------------
    # Placeholder: Implicit CBF  h_I(x)  (eq. 5)
    # ------------------------------------------------------------------

    def compute_implicit_cbf(self, robot_state, human_states, tbc, tau_0):
        """
        Compute the implicit control barrier function value:

            h_I(x) = min_{t in [0,T]} { h(Phi_pi(x,t)), h_B(Phi_pi(x,T)) }

        This requires simulating the flow of the system under the TBC
        from the current state and evaluating the safety / backup-set
        constraints at each propagated point.

        Parameters
        ----------
        robot_state  : FullState
        human_states : list[ObservableState]
        tbc          : TimeVaryingBackupController
        tau_0        : float — current time-offset

        Returns
        -------
        h_I : float — implicit CBF value (>= 0 means safe)
        """
        traj = self.simulate_flow(robot_state, human_states, tbc, tau_0, self.T)

        h_min = float("inf")
        for k, row in enumerate(traj):
            proxy = FullState(
                px=row[0],
                py=row[1],
                vx=row[2],
                vy=row[3],
                radius=robot_state.radius,
                gx=robot_state.gx,
                gy=robot_state.gy,
                v_pref=robot_state.v_pref,
                theta=robot_state.theta,
            )
            # Safe-set constraints at every point
            h_vals = self.safety.h_safe(proxy, human_states)
            if h_vals:
                h_min = min(h_min, min(h_vals))

            # Backup-set constraint at the terminal point only
            if k == len(traj) - 1:
                h_min = min(h_min, self.safety.h_backup(proxy))

        return h_min if h_min < float("inf") else 1.0

    # ------------------------------------------------------------------
    # Feasibility check  (shared by Algorithm 1 and Algorithm 2)
    # ------------------------------------------------------------------

    def _tbc_is_feasible(self, traj: np.ndarray) -> bool:
        """
        Return True iff the trajectory under `tbc` from `robot_state`:
          (a) stays entirely inside the safe set S  (h_safe >= 0 at every step), and
          (b) ends inside the backup set S_B         (h_backup >= 0 at terminal step).

        This is the feasibility predicate used in both Algorithm 1 and
        Algorithm 2 of the paper.

        Parameters
        ----------
        traj : np.ndarray, shape (n_steps + 1, n_dims)

        Returns
        -------
        bool
        """

        # For every state in the trajectory, check if it is in the safe set
        for state in traj:
            if not self.safety.h_safe(state, []):
                return False

        # Check if the terminal state is in the backup set
        if not self.safety.h_backup(traj[-1]):
            return False

        return True

    # ------------------------------------------------------------------
    # Dynamics-agnostic ODE helpers  (mirrors dynamics.py structure)
    # ------------------------------------------------------------------

    def f_x(self, x):
        """
        Drift term of the control-affine dynamics  ẋ = f(x) + g(x)u.

        For the 2-D single integrator:  f(x) = A x = [0, 0]  (no drift).
        """
        return self.A @ x

    def g_x(self, x):
        """
        Input matrix of the control-affine dynamics  ẋ = f(x) + g(x)u.

        For the 2-D single integrator:  g(x) = B = I  (identity, constant).
        """
        return self.B

    def _prop_main(self, t, x, u):
        """
        ODE right-hand side:  ẋ = f(x) + g(x) u.

        Mirrors Dynamics.propMain from dynamics.py but for the 2-D
        single-integrator  (ṗ = v,  state = [px, py],  u = [vx, vy]).
        """
        return self.f_x(x) + self.g_x(x) @ u

    def integrateState(
        self,
        x: np.ndarray,
        u: np.ndarray,
        t_span: np.ndarray,
        dist: np.ndarray,
        options: dict,
    ) -> np.ndarray:
        """
        Advance position state x by one time step dt under constant
        velocity command u, using RK45 via solve_ivp.

        Mirrors Dynamics.integrateState from dynamics.py.

        Parameters
        ----------
        x  : np.ndarray, shape (2,)  — [px, py]
        u  : np.ndarray, shape (2,)  — [vx, vy]  (velocity command)
        t_span : np.ndarray
        dist : np.ndarray, shape (2,)  — [dx, dy]
        options : dict

        Returns
        -------
        x_next : np.ndarray, shape (2,)  — [px_next, py_next]
        """
        t_step = (0.0, t_span[-1])
        sol = solve_ivp(
            lambda t, x: self._prop_main(t, x, u),
            t_step,
            x,
            method="RK45",
            rtol=self.int_options["rtol"],
            atol=self.int_options["atol"],
            t_eval=t_span,
        )
        return sol.y[:, -1]

    # ------------------------------------------------------------------
    # Flow simulation
    # ------------------------------------------------------------------

    def simulate_flow(
        self,
        robot_state: FullState,
        tbc: TimeVaryingBackupController,
        tau_0: float,
        T: float,
    ) -> np.ndarray:
        """
        Simulate system state forward using the time varying backup controller to select an action for the robot and propagate the robot's dynamics forward for t in [0, T] with step self.dt, using
        RK45 integration of the 4-D single-integrator dynamics  (ṗ = v).

        Parameters
        ----------
        robot_state  : FullState | ObservableState — initial state
        tbc          : TimeVaryingBackupController — active TBC (pi)
        tau_0        : float — current time-offset
        T            : float — horizon

        Returns
        -------
        trajectory : np.ndarray, shape (n_steps + 1, 2)
            Each row is [px, py, vx, vy].
        """

        trajectory = []
        t_span = np.arange(0, T, self.dt)

        for t in t_span:
            # 1) calculate robot action
            action = tbc.evaluate(tau_0 + t, robot_state, human_states=None)

            # 2) propagate robot state forward using the action
            x = np.array([robot_state.px, robot_state.py])
            new_x = self.integrateState(
                x, action, t_span, dist=np.array([0.0, 0.0]), options=self.int_options
            )
            trajectory.append(new_x)

        return np.array(trajectory, dtype=float).reshape(-1, 2)

    # ------------------------------------------------------------------
    # Regulation function  (eq. 7-8)  — fully implemented
    # ------------------------------------------------------------------

    def regulation_function(self, u_des, u_backup, h_I):
        """
        Blend desired and backup actions via the regulation function:

            lambda = 1 - exp(-beta * max(0, h_I))
            u_act  = lambda * u_des + (1 - lambda) * u_backup

        When h_I >> 0 (far from boundary), lambda -> 1  =>  follow desired.
        When h_I -> 0 (at boundary),       lambda -> 0  =>  follow backup.
        """
        lam = 1.0 - np.exp(-self.beta * max(0.0, h_I))
        return lam * u_des + (1.0 - lam) * u_backup

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_desired_action(self, state):
        """Get action from nominal policy as a numpy array."""
        if self.nominal_policy is not None:
            self.nominal_policy.time_step = self.time_step
            action = self.nominal_policy.predict(state)
        else:
            action = self._goal_seeking_action(state)

        if isinstance(action, ActionXY):
            return np.array([action.vx, action.vy])
        elif isinstance(action, ActionRot):
            return np.array([action.v, action.r])
        return np.array(action, dtype=float)

    def _goal_seeking_action(self, state):
        """Fallback: simple move-toward-goal when no nominal policy is set."""
        s = state.self_state
        dx = s.gx - s.px
        dy = s.gy - s.py
        d = norm([dx, dy])
        if d < 1e-3:
            return ActionXY(0, 0)
        return ActionXY(s.v_pref * dx / d, s.v_pref * dy / d)

    def reset(self):
        """Reset internal state between episodes."""
        self.tau_0 = 0.0
        self.system_time = 0.0
        self.active_tbc_index = 0
        for tbc in self.tbcs:
            if isinstance(tbc.maneuver, CarryOnManeuver):
                tbc.maneuver.held_action = np.array([0.0, 0.0])


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from crowd_sim.envs.utils.state import FullState, ObservableState, JointState

    # -- Build maneuvers ----------
    maneuvers = [EvadeManeuver()]

    # -- Build a TVBCBF policy with default evade TBCs ----------
    policy = TVBCBF()
    policy.kinematics = "holonomic"
    policy.time_step = 0.25
    policy.build_default_tbcs(
        maneuvers=maneuvers, backup_mode="stop", T_M=0.5, delta=0.2
    )

    print(f"Policy: {policy.name}")
    print(f"Registered TBCs: {policy.tbcs}")

    # -- Fake a robot heading toward a goal with one human nearby ----------
    robot = FullState(
        px=0.0,
        py=0.0,
        vx=0.5,
        vy=0.0,
        radius=0.1,
        gx=10.0,
        gy=0.0,
        v_pref=1.0,
        theta=0.0,
    )
    human = ObservableState(px=3.0, py=0.0, vx=0.0, vy=0.0, radius=0.1)
    state = JointState(robot, [human])

    # -- Test tbc feasibility
    tbc = policy.tbcs[0]
    print(f"TBC: {tbc}")
    feasibility = policy._tbc_is_feasible(robot, [human], tbc, tau_0=-2.0)
    print(f"TBC is feasible: {feasibility}")

    # -- Step a few iterations ---------------------------------------------
    n_steps = 8
    print(f"\n{'step':>4}  {'action':>20}  {'tau_0':>8}  {'tbc':>10}")
    print("-" * 52)

    for i in range(n_steps):
        action = policy.predict(state)
        tbc_name = policy.tbcs[policy.active_tbc_index].name
        print(
            f"{i:4d}  ({action.vx:+.3f}, {action.vy:+.3f})  "
            f"{policy.tau_0:+8.3f}  {tbc_name:>10}"
        )

        # Advance the fake robot state with the chosen action
        robot = FullState(
            px=robot.px + action.vx * policy.time_step,
            py=robot.py + action.vy * policy.time_step,
            vx=action.vx,
            vy=action.vy,
            radius=robot.radius,
            gx=robot.gx,
            gy=robot.gy,
            v_pref=robot.v_pref,
            theta=robot.theta,
        )
        # Human walks forward at constant velocity
        human = ObservableState(
            px=human.px + human.vx * policy.time_step,
            py=human.py + human.vy * policy.time_step,
            vx=human.vx,
            vy=human.vy,
            radius=human.radius,
        )
        state = JointState(robot, [human])

    print(
        "\nDone. The placeholders (compute_time_offset, switch_maneuver, "
        "compute_implicit_cbf)\nreturn defaults — replace them with real logic."
    )
