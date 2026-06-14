"""
Time-Varying Backup Control Barrier Function (TVBCBF) Policy for a Robotic Manipulator.

Dynamics-agnostic safety filter following:
    Singletary et al., "Safe Drone Flight with Time-Varying Backup Controllers,"
    IEEE/RSJ IROS 2022.

The user supplies their own dynamics by subclassing TVBCBF and implementing
f_x(x) and g_x(x) for control-affine dynamics  ẋ = f(x) + g(x) u, along with
concrete BackupManeuver, BackupController, and SafetyConstraints subclasses.
"""

import abc
import logging
import math
import numpy as np
from numpy.linalg import norm
from scipy.integrate import solve_ivp

from typing import Optional, List, Tuple, Sequence


# ---------------------------------------------------------------------------
# Backup Maneuver
# ---------------------------------------------------------------------------


class BackupManeuver(abc.ABC):
    """Maneuver uM(x) executed for duration T_M before backup engages."""

    def __init__(self, name: str = "maneuver"):
        self.name = name

    @abc.abstractmethod
    def compute_action(self, x: np.ndarray, human_states: List, **kwargs) -> np.ndarray:
        """Return the maneuver action u as np.ndarray."""

    def __repr__(self):
        return f"BackupManeuver({self.name})"


# ---------------------------------------------------------------------------
# Backup Controller (uB)
# ---------------------------------------------------------------------------


class BackupController(abc.ABC):
    """Terminal backup controller uB(x) that drives the system into S_B."""

    @abc.abstractmethod
    def compute_action(self, x: np.ndarray, human_states: List) -> np.ndarray:
        """Return the backup action u as np.ndarray."""


# ---------------------------------------------------------------------------
# Time-Varying Backup Controller  (eq. 9)
# ---------------------------------------------------------------------------


class TimeVaryingBackupController:
    """Composite policy pi(x, tau): maneuver -> transition -> backup."""

    def __init__(
        self,
        maneuver: BackupManeuver,
        backup: BackupController,
        T_M: float = 0.5,
        delta: float = 0.2,
    ):
        self.maneuver = maneuver
        self.backup = backup
        self.T_M = T_M
        self.delta = delta

    @property
    def name(self):
        return self.maneuver.name

    def evaluate(
        self,
        relative_time: float,
        x: np.ndarray,
        human_states: Optional[List] = None,
        **kwargs,
    ) -> np.ndarray:
        if human_states is None:
            human_states = []
        u_m = np.asarray(
            self.maneuver.compute_action(x, human_states, **kwargs), dtype=float
        )
        u_b = np.asarray(self.backup.compute_action(x, human_states), dtype=float)

        if relative_time <= self.T_M:
            return u_m
        elif relative_time <= self.T_M + self.delta:
            alpha = (relative_time - self.T_M) / self.delta
            return (1.0 - alpha) * u_m + alpha * u_b
        else:
            return u_b

    def get_backup_action(self, x: np.ndarray, human_states: List) -> np.ndarray:
        return np.asarray(self.backup.compute_action(x, human_states), dtype=float)

    def __repr__(self):
        return (
            f"TBC(maneuver={self.maneuver.name}, "
            f"T_M={self.T_M}, delta={self.delta})"
        )


# ---------------------------------------------------------------------------
# Safety Primitives (safe set S, backup set S_B)
# ---------------------------------------------------------------------------


class SafetyConstraints(abc.ABC):
    """Evaluates h(x) >= 0 (safe set) and hB(x) >= 0 (backup set)."""

    @abc.abstractmethod
    def h_safe(self, x: np.ndarray, human_states: List) -> List[float]:
        """Per-obstacle safety values; safe iff every entry is >= 0."""

    @abc.abstractmethod
    def h_backup(self, x: np.ndarray) -> float:
        """Backup-set value; satisfied iff >= 0."""

    def in_safe_set(
        self, x: np.ndarray, human_states: List, epsilon_t: float = 0.0
    ) -> bool:
        return all(v >= epsilon_t for v in self.h_safe(x, human_states))

    def in_backup_set(self, x: np.ndarray, epsilon_b: float = 0.0) -> bool:
        return self.h_backup(x) >= epsilon_b


# ---------------------------------------------------------------------------
# TVBCBF Policy
# ---------------------------------------------------------------------------


class TVBCBF(abc.ABC):
    """
    Time-Varying Backup CBF safety filter.

    Subclass and implement f_x(x) and g_x(x). Attach a SafetyConstraints
    subclass via `self.safety` and a TimeVaryingBackupController via
    `set_tbc()`. The nominal (desired) policy must be a callable
    `(x, human_states) -> np.ndarray` (or have a matching `predict` method).
    """

    def __init__(self):
        self.name = "TVBCBF"
        self.int_options = {"rtol": 1e-6, "atol": 1e-6}

        self.nominal_policy = None

        self.tbc: Optional[TimeVaryingBackupController] = None

        self.delta_tau = 0.1
        self.system_time = 0.0
        self.global_time = 0.0
        self.tau_0 = 0.0
        self.relative_time = 0.0
        self.system_times = []
        self.tau_0s = []
        self.relative_times = []

        self.beta = 3.0
        self.lambdas = []
        self.h_Is = []
        self.h_safe_mins = []
        self.h_backups = []

        self.T = 2.0
        self.dt = 0.25
        self.time_step = self.dt

        self.backup_trajectories = []
        self.actions = []

        self.safety: Optional[SafetyConstraints] = None
        self.epsilon_tau: Sequence[float] = [0.0]
        self.epsilon_backup: float = 0.0

        self.full_backup = True

    # ------------------------------------------------------------------
    # Configuration & setup
    # ------------------------------------------------------------------

    def configure(self, config):
        if config.has_section("tvbcbf"):
            self.beta = config.getfloat("tvbcbf", "beta")
            self.T = config.getfloat("tvbcbf", "backup_horizon")
            self.dt = config.getfloat("tvbcbf", "dt")

    def set_nominal_policy(self, policy):
        self.nominal_policy = policy

    def set_tbc(self, tbc: TimeVaryingBackupController):
        self.tbc = tbc
        logging.info("TVBCBF: set TBC %s", tbc)

    # ------------------------------------------------------------------
    # Forwarding hooks to nominal policy
    # ------------------------------------------------------------------

    def set_phase(self, phase):
        self.phase = phase
        if self.nominal_policy is not None and hasattr(
            self.nominal_policy, "set_phase"
        ):
            self.nominal_policy.set_phase(phase)

    def set_device(self, device):
        self.device = device
        if self.nominal_policy is not None and hasattr(
            self.nominal_policy, "set_device"
        ):
            self.nominal_policy.set_device(device)

    def set_env(self, env):
        self.env = env
        if self.nominal_policy is not None and hasattr(self.nominal_policy, "set_env"):
            self.nominal_policy.set_env(env)

    # ------------------------------------------------------------------
    # Core predict loop
    # ------------------------------------------------------------------

    def predict(self, x: np.ndarray, human_states: Optional[List] = None) -> np.ndarray:
        if self.tbc is None:
            raise RuntimeError(
                "TVBCBF: no backup controller registered. Call set_tbc() first."
            )
        if self.safety is None:
            raise RuntimeError(
                "TVBCBF: no safety constraints registered. Set self.safety first."
            )
        if human_states is None:
            human_states = []

        x = np.asarray(x, dtype=float)
        tbc = self.tbc

        # 1) Desired action from nominal policy
        u_des = self._get_desired_action(x, human_states)

        self.epsilon_tau, self.epsilon_backup = [0.0], 0.0

        # 2) Update time-offset (Algorithm 1)
        self.tau_0, backup_trajectory, h_safe_vals, h_backup_val = (
            self.compute_time_offset(
                x=x,
                human_states=human_states,
                global_time=self.global_time,
                tau_0_prev=self.tau_0,
                tbc=tbc,
                dt=self.dt,
            )
        )

        if self.full_backup:
            self.tau_0 = 0.0

        self.backup_trajectories.append(np.array(backup_trajectory))

        self.tau_0s.append(self.tau_0)
        self.system_times.append(self.system_time)
        self.relative_times.append(self.system_time - self.tau_0)

        # 3) Implicit CBF value h_I(x)
        h_I = self.compute_implicit_cbf(
            backup_trajectory, human_states, h_safe_vals, h_backup_val
        )
        self.h_Is.append(h_I)
        self.h_safe_mins.append(
            float(np.min(h_safe_vals)) if len(h_safe_vals) > 0 else np.inf
        )
        self.h_backups.append(float(np.min(h_backup_val)))

        # 4) Regulation function (eq. 7-8)
        u_backup = tbc.evaluate(self.global_time - self.tau_0, x, human_states)
        u_act = self.regulation_function(u_des, u_backup, h_I)

        self.actions.append(np.asarray(u_act, dtype=float))

        self.global_time += self.dt

        return u_act

    # ------------------------------------------------------------------
    # Time-offset computation (Algorithm 1)
    # ------------------------------------------------------------------

    def compute_time_offset(
        self,
        x: np.ndarray,
        human_states: List,
        global_time: float,
        tau_0_prev: float,
        tbc: TimeVaryingBackupController,
        dt: float,
    ) -> Tuple[float, List[np.ndarray], List[np.ndarray], float]:
        if self.full_backup:
            tau_0 = 0.0
        else:
            tau_0 = global_time

        traj = self.simulate_flow(
            x=x,
            tbc=tbc,
            global_time=global_time,
            tau_0=tau_0,
            T=self.T,
            human_state=human_states[0] if len(human_states) > 0 else None,
        )

        safe_bool, h_safe_vals, h_backup_val = self._tbc_is_feasible(
            traj,
            human_states,
            epsilon_tau=self.epsilon_tau,
            epsilon_backup=self.epsilon_backup,
        )
        if safe_bool:
            return tau_0, traj, h_safe_vals, h_backup_val

        tau_0 = tau_0_prev - self.dt
        new_traj = self.simulate_flow(
            x=x,
            tbc=tbc,
            global_time=global_time,
            tau_0=tau_0,
            T=self.T,
            human_state=human_states[0] if len(human_states) > 0 else None,
        )
        safe_bool, h_safe_vals, h_backup_val = self._tbc_is_feasible(
            new_traj,
            human_states,
            epsilon_tau=self.epsilon_tau,
            epsilon_backup=self.epsilon_backup,
        )
        return tau_0, new_traj, h_safe_vals, h_backup_val

    # ------------------------------------------------------------------
    # Implicit CBF h_I(x) (eq. 5)
    # ------------------------------------------------------------------

    def compute_implicit_cbf(
        self,
        backup_trajectory: List[np.ndarray],
        human_states: List,
        h_vals: List[np.ndarray],
        h_backup_val: float,
    ) -> float:
        h_vals_min = np.min(h_vals) if len(h_vals) > 0 else np.inf
        return min(h_vals_min, h_backup_val)

    # ------------------------------------------------------------------
    # Feasibility check
    # ------------------------------------------------------------------

    def _tbc_is_feasible(
        self,
        traj: List[np.ndarray],
        human_states: List,
        epsilon_tau: float | Sequence[float] = 0.0,
        epsilon_backup: float = 0.0,
    ) -> Tuple[bool, List[np.ndarray], float]:
        num_steps = len(traj)
        num_humans = len(human_states)
        h_safe_vals = [np.zeros(num_steps, dtype=float) for _ in range(num_humans)]

        for step_idx, state in enumerate(traj):
            step_h_safe_vals = self.safety.h_safe(state, human_states)
            for human_idx in range(num_humans):
                h_safe_vals[human_idx][step_idx] = step_h_safe_vals[human_idx]

        h_backup_val = self.safety.h_backup(traj[-1])

        for step_idx in range(num_steps - 1):
            eps_tau = 0.0
            if num_humans == 0:
                step_min_h = np.inf
            else:
                step_min_h = min(
                    h_safe_vals[human_idx][step_idx] for human_idx in range(num_humans)
                )
            if step_min_h < eps_tau:
                return False, h_safe_vals, h_backup_val

        if h_backup_val < epsilon_backup:
            return False, h_safe_vals, h_backup_val
        return True, h_safe_vals, h_backup_val

    # ------------------------------------------------------------------
    # Dynamics (user-supplied via subclass)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def f_x(self, x: np.ndarray) -> np.ndarray:
        """Drift term f(x) of the control-affine dynamics ẋ = f(x) + g(x)u."""

    @abc.abstractmethod
    def g_x(self, x: np.ndarray) -> np.ndarray:
        """Input matrix g(x) of the control-affine dynamics ẋ = f(x) + g(x)u."""

    def _prop_main(self, t, x, u):
        return self.f_x(x) + self.g_x(x) @ u

    def integrateState(
        self,
        x: np.ndarray,
        u: np.ndarray,
        t_step: Sequence[float],
        options: dict,
    ) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)

        sol = solve_ivp(
            lambda t, x: self._prop_main(t, x, u),
            t_step,
            x,
            method="RK45",
            rtol=self.int_options["rtol"],
            atol=self.int_options["atol"],
            t_eval=t_step,
        )
        return sol.y[:, -1]

    # ------------------------------------------------------------------
    # Flow simulation
    # ------------------------------------------------------------------

    def simulate_flow(
        self,
        x: np.ndarray,
        tbc: TimeVaryingBackupController,
        global_time: float,
        tau_0: float,
        T: float,
        human_state=None,
    ) -> List[np.ndarray]:
        x = np.asarray(x, dtype=float).copy()
        trajectory = [x.copy()]
        N = int(math.ceil(T / self.dt))

        for t in range(N):
            action = tbc.evaluate(
                self.system_time + t * self.dt - tau_0, x, human_states=None
            )
            new_x = self.integrateState(
                x, action, t_step=[0, self.dt], options=self.int_options
            )
            x = new_x
            trajectory.append(x.copy())

        return trajectory

    # ------------------------------------------------------------------
    # Regulation function (eq. 7-8)
    # ------------------------------------------------------------------

    def regulation_function(
        self,
        u_des: np.ndarray,
        u_backup: np.ndarray,
        h_I: float,
    ) -> np.ndarray:
        u_des_act = np.asarray(u_des, dtype=float)
        u_backup_act = np.asarray(u_backup, dtype=float)

        lam = 1.0 - np.exp(-self.beta * max(0.0, h_I))
        lam = float(np.clip(lam, 0.0, 1.0))

        if self.full_backup:
            lam = 0.0

        u_act = lam * u_des_act + (1.0 - lam) * u_backup_act

        self.lambdas.append(lam)
        return u_act

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_desired_action(self, x: np.ndarray, human_states: List) -> np.ndarray:
        if self.nominal_policy is None:
            raise RuntimeError(
                "TVBCBF: no nominal policy set. Call set_nominal_policy() first."
            )
        if hasattr(self.nominal_policy, "predict"):
            action = self.nominal_policy.predict(x, human_states)
        else:
            action = self.nominal_policy(x, human_states)
        return np.asarray(action, dtype=float)

    def reset(self):
        self.tau_0 = 0.0
        self.system_time = 0.0
        self.global_time = 0.0
        self.system_times = []
        self.tau_0s = []
        self.relative_times = []
        self.h_Is = []
        self.h_safe_mins = []
        self.h_backups = []
        self.lambdas = []
        self.actions = []
        self.backup_trajectories = []


# ---------------------------------------------------------------------------
# Skeleton — plug in your manipulator-specific code below
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    class MyManeuver(BackupManeuver):
        def compute_action(self, x, human_states, **kwargs):
            raise NotImplementedError

    class MyBackupController(BackupController):
        def compute_action(self, x, human_states):
            raise NotImplementedError

    class MySafety(SafetyConstraints):
        def h_safe(self, x, human_states):
            raise NotImplementedError

        def h_backup(self, x):
            raise NotImplementedError

    class MyManipulatorTVBCBF(TVBCBF):
        def f_x(self, x):
            raise NotImplementedError

        def g_x(self, x):
            raise NotImplementedError

    policy = MyManipulatorTVBCBF()
    policy.safety = MySafety()
    policy.set_tbc(
        TimeVaryingBackupController(
            MyManeuver(), MyBackupController(), T_M=0.5, delta=0.2
        )
    )
    # policy.set_nominal_policy(my_nominal_policy)
    # x = np.zeros(n)
    # u = policy.predict(x, human_states=[])
