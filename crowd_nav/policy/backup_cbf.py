import numpy as np
import cvxopt
from numpy.linalg import norm
import logging
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.state import ObservableState, FullState


class BackupCBF(Policy):
    """
    Backup Control Barrier Function Policy

    This policy uses a backup controller to define a safe set and filters
    nominal policy actions through a CBF-QP to ensure safety.

    The backup controller is a simple controller that can always bring the
    system to a safe state (e.g., stop or move away from obstacles).
    """

    def __init__(self):
        super().__init__()
        self.name = "BackupCBF"
        self.trainable = False
        self.kinematics = None
        self.time_step = None

        # Backup CBF parameters
        self.cbf_rate = None  # gamma - CBF rate parameter
        self.backup_time_horizon = None  # T - time horizon for backup controller
        self.safety_radius = None  # minimum safe distance

        # Nominal policy (backup policy to use)
        self.nominal_policy = None
        self.nominal_policy_name = None

        # Backup controller parameters
        self.backup_controller_type = None  # 'stop' or 'retreat'
        self.backup_controller_gain = None

    def configure(self, config):
        """
        Configure the backup CBF policy from config file

        Expected config sections:
        - backup_cbf: cbf_rate, backup_time_horizon, safety_radius,
                      backup_controller_type, backup_controller_gain
        - action_space: kinematics, time_step
        """
        self.kinematics = config.get("action_space", "kinematics")
        self.time_step = config.getfloat("action_space", "time_step")

        # Backup CBF parameters
        self.cbf_rate = config.getfloat("backup_cbf", "cbf_rate")
        self.backup_time_horizon = config.getfloat("backup_cbf", "backup_time_horizon")
        self.safety_radius = config.getfloat("backup_cbf", "safety_radius")
        self.backup_controller_type = config.get("backup_cbf", "backup_controller_type")
        self.backup_controller_gain = config.getfloat(
            "backup_cbf", "backup_controller_gain"
        )

        # Nominal policy name (if using another policy as nominal)
        try:
            self.nominal_policy_name = config.get("backup_cbf", "nominal_policy")
        except:
            self.nominal_policy_name = None

        logging.info("Policy: BackupCBF with {} kinematics".format(self.kinematics))
        logging.info(
            "CBF rate: {}, Backup horizon: {}, Safety radius: {}".format(
                self.cbf_rate, self.backup_time_horizon, self.safety_radius
            )
        )

    def set_nominal_policy(self, policy):
        """Set the nominal policy to use (e.g., CADRL, SARL, etc.)"""
        self.nominal_policy = policy

    def predict(self, state):
        """
        Predict action using backup CBF

        Steps:
        1. Get nominal action from nominal policy (or use zero action)
        2. For each human obstacle, compute backup CBF constraint
        3. Solve QP to find safe action closest to nominal action
        4. Return filtered safe action
        """
        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == "holonomic" else ActionRot(0, 0)

        # Get nominal action
        if self.nominal_policy is not None:
            nominal_action = self.nominal_policy.predict(state)
        else:
            # Default: move towards goal
            nominal_action = self._get_nominal_action(state)

        # Convert nominal action to control vector
        if self.kinematics == "holonomic":
            u_nom = np.array([nominal_action.vx, nominal_action.vy])
        else:
            u_nom = np.array([nominal_action.v, nominal_action.r])

        # Filter action through backup CBF-QP
        safe_action = self._backup_cbf_filter(state, u_nom)

        # Convert back to action type
        if self.kinematics == "holonomic":
            return ActionXY(safe_action[0], safe_action[1])
        else:
            return ActionRot(safe_action[0], safe_action[1])

    def _get_nominal_action(self, state):
        """Get a simple nominal action (move towards goal)"""
        self_state = state.self_state
        dx = self_state.gx - self_state.px
        dy = self_state.gy - self_state.py
        dist = norm([dx, dy])

        if dist < 0.1:
            if self.kinematics == "holonomic":
                return ActionXY(0, 0)
            else:
                return ActionRot(0, 0)

        # Normalize direction and scale by preferred velocity
        vx = (dx / dist) * self_state.v_pref
        vy = (dy / dist) * self_state.v_pref

        if self.kinematics == "holonomic":
            return ActionXY(vx, vy)
        else:
            # For unicycle, compute desired heading
            desired_theta = np.arctan2(dy, dx)
            theta_error = desired_theta - self_state.theta
            # Normalize angle to [-pi, pi]
            theta_error = np.arctan2(np.sin(theta_error), np.cos(theta_error))
            v = self_state.v_pref
            r = self.backup_controller_gain * theta_error
            return ActionRot(v, r)

    def _backup_cbf_filter(self, state, u_nom):
        """
        Filter nominal action through backup CBF-QP

        For each human obstacle, we enforce:
        h_dot + gamma * h >= 0

        where h is the backup CBF value computed from the backup controller flow.
        """
        self_state = state.self_state

        if len(state.human_states) == 0:
            # No obstacles, return nominal action
            return u_nom

        # Set up QP: minimize ||u - u_nom||^2 subject to CBF constraints
        n = len(state.human_states)

        if self.kinematics == "holonomic":
            action_dim = 2
        else:
            action_dim = 2

        # QP: minimize 0.5 * u^T * P * u + q^T * u
        # subject to G * u <= h

        P = cvxopt.matrix(np.eye(action_dim))
        q = cvxopt.matrix(-u_nom)

        # Collect constraints from all humans
        G_list = []
        h_list = []

        for human_state in state.human_states:
            # Compute backup CBF constraint for this human
            G_h, h_h = self._compute_backup_cbf_constraint(self_state, human_state)
            if G_h is not None and h_h is not None:
                G_list.append(G_h)
                h_list.append(h_h)

        if len(G_list) == 0:
            # No active constraints, return nominal action
            return u_nom

        # Stack constraints
        G = cvxopt.matrix(np.vstack(G_list))
        h = cvxopt.matrix(np.hstack(h_list))

        # Add input constraints (velocity limits)
        if self.kinematics == "holonomic":
            # |vx| <= v_pref, |vy| <= v_pref
            v_max = self_state.v_pref
            G_input = np.array(
                [
                    [1, 0],  # vx <= v_max
                    [-1, 0],  # -vx <= v_max
                    [0, 1],  # vy <= v_max
                    [0, -1],  # -vy <= v_max
                ]
            )
            h_input = np.array([v_max, v_max, v_max, v_max])
        else:
            # v <= v_pref, |r| <= r_max
            v_max = self_state.v_pref
            r_max = np.pi / 4  # max angular velocity
            G_input = np.array(
                [
                    [1, 0],  # v <= v_max
                    [-1, 0],  # -v <= 0 (v >= 0)
                    [0, 1],  # r <= r_max
                    [0, -1],  # -r <= r_max
                ]
            )
            h_input = np.array([v_max, 0, r_max, r_max])

        # Combine CBF constraints with input constraints
        G_all = np.vstack([G, G_input])
        h_all = np.hstack([h, h_input])
        G = cvxopt.matrix(G_all)
        h = cvxopt.matrix(h_all)

        # Solve QP
        try:
            cvxopt.solvers.options["show_progress"] = False
            solution = cvxopt.solvers.qp(P, q, G, h)

            if solution["status"] == "optimal":
                u_safe = np.array(solution["x"]).flatten()
            else:
                # QP infeasible, use backup controller
                logging.warning("Backup CBF QP infeasible, using backup controller")
                u_safe = self._backup_controller(state)
        except:
            # QP failed, use backup controller
            logging.warning("Backup CBF QP failed, using backup controller")
            u_safe = self._backup_controller(state)

        return u_safe

    def _compute_backup_cbf_constraint(self, robot_state, human_state):
        """
        Compute backup CBF constraint: h_dot + gamma * h >= 0

        The backup CBF h(x) is defined based on the flow of the system
        under the backup controller. We check if applying the backup controller
        for time T keeps the system safe.

        Returns: (G, h) where G*u <= h is the constraint
        """
        # Compute relative position and velocity
        p_rel = np.array(
            [robot_state.px - human_state.px, robot_state.py - human_state.py]
        )
        v_rel = np.array(
            [robot_state.vx - human_state.vx, robot_state.vy - human_state.vy]
        )

        # Distance between centers
        d = norm(p_rel)
        if d < 1e-6:
            d = 1e-6

        # Minimum safe distance (sum of radii)
        d_safe = robot_state.radius + human_state.radius + self.safety_radius

        # Backup CBF: h(x) = d - d_safe
        h = d - d_safe

        # Compute backup controller action
        u_backup = self._backup_controller_for_human(robot_state, human_state)

        # Compute system dynamics
        if self.kinematics == "holonomic":
            # x_dot = u (position directly controlled)
            # For holonomic, we assume position dynamics: p_dot = v
            # So we need velocity dynamics: v_dot = u (acceleration control)
            # But in this codebase, actions are velocities, so p_dot = u
            # We'll approximate: h_dot = dh/dp * p_dot = dh/dp * u

            # Gradient of h with respect to robot position
            dh_dp = p_rel / d  # unit vector from human to robot

            # Under backup controller: p_dot_backup = u_backup
            # Under current action: p_dot = u
            # h_dot = dh/dp * (p_dot - p_dot_backup) + dh/dp * p_dot_backup
            #       = dh/dp * u - dh/dp * u_backup + dh/dp * u_backup
            #       = dh/dp * u

            # Actually, for holonomic with velocity control:
            # p_dot = v (current velocity)
            # v_dot = u (action is acceleration)
            # But the codebase uses velocity actions, so we approximate:
            # h_dot ≈ dh/dp * (u - v_human) where u is robot velocity action

            # More accurate: h_dot = dh/dp * (v_robot - v_human)
            # With action u: v_robot_new = u
            # So h_dot = dh/dp * (u - v_human)

            v_human = np.array([human_state.vx, human_state.vy])
            h_dot_nom = np.dot(dh_dp, u_backup - v_human)

            # Constraint: h_dot + gamma * h >= 0
            # dh/dp * (u - v_human) + gamma * h >= 0
            # dh/dp * u >= -gamma * h + dh/dp * v_human

            G = -dh_dp.reshape(1, -1)  # Note: negative because we want >=
            h_constraint = self.cbf_rate * h - np.dot(dh_dp, v_human)

        else:
            # Unicycle dynamics
            # p_dot = [v*cos(theta), v*sin(theta)]
            # theta_dot = r

            # Gradient of h
            dh_dp = p_rel / d

            # Under backup controller
            u_backup_v, u_backup_r = u_backup[0], u_backup[1]
            theta_backup = robot_state.theta + u_backup_r * self.backup_time_horizon
            p_dot_backup = np.array(
                [u_backup_v * np.cos(theta_backup), u_backup_v * np.sin(theta_backup)]
            )

            # Linearization around current state
            # h_dot ≈ dh/dp * p_dot
            # p_dot = [v*cos(theta), v*sin(theta)]
            # For small changes: p_dot ≈ [v*cos(theta0), v*sin(theta0)] + ...

            # Simplified: assume current theta
            theta = robot_state.theta
            dh_dv = dh_dp * np.array([np.cos(theta), np.sin(theta)])
            dh_dr = dh_dp * np.array(
                [-u_backup_v * np.sin(theta), u_backup_v * np.cos(theta)]
            )

            # Constraint: h_dot + gamma * h >= 0
            # Approximate h_dot ≈ dh_dv * v + dh_dr * r
            G = -np.array([dh_dv, dh_dr]).reshape(1, -1)
            h_dot_backup = np.dot(
                dh_dp, p_dot_backup - np.array([human_state.vx, human_state.vy])
            )
            h_constraint = self.cbf_rate * h - h_dot_backup

        return G, np.array([h_constraint])

    def _backup_controller(self, state):
        """
        Backup controller: simple controller that ensures safety
        Returns control action
        """
        self_state = state.self_state

        if len(state.human_states) == 0:
            # No obstacles, stop
            if self.kinematics == "holonomic":
                return np.array([0.0, 0.0])
            else:
                return np.array([0.0, 0.0])

        # Find nearest human
        min_dist = float("inf")
        nearest_human = None
        for human_state in state.human_states:
            dist = norm(
                [self_state.px - human_state.px, self_state.py - human_state.py]
            )
            if dist < min_dist:
                min_dist = dist
                nearest_human = human_state

        return self._backup_controller_for_human(self_state, nearest_human)

    def _backup_controller_for_human(self, robot_state, human_state):
        """
        Backup controller for a specific human obstacle

        Options:
        - 'stop': stop moving
        - 'retreat': move away from human
        """
        if self.backup_controller_type == "stop":
            if self.kinematics == "holonomic":
                return np.array([0.0, 0.0])
            else:
                return np.array([0.0, 0.0])

        elif self.backup_controller_type == "retreat":
            # Move away from human
            p_rel = np.array(
                [robot_state.px - human_state.px, robot_state.py - human_state.py]
            )
            d = norm(p_rel)
            if d < 1e-6:
                # Too close, stop
                if self.kinematics == "holonomic":
                    return np.array([0.0, 0.0])
                else:
                    return np.array([0.0, 0.0])

            # Direction away from human
            direction = p_rel / d

            if self.kinematics == "holonomic":
                # Move away with some velocity
                v_retreat = min(robot_state.v_pref, self.backup_controller_gain * d)
                return v_retreat * direction
            else:
                # For unicycle, align heading with retreat direction
                desired_theta = np.arctan2(direction[1], direction[0])
                theta_error = desired_theta - robot_state.theta
                theta_error = np.arctan2(np.sin(theta_error), np.cos(theta_error))

                v_retreat = min(robot_state.v_pref, self.backup_controller_gain * d)
                r = self.backup_controller_gain * theta_error
                return np.array([v_retreat, r])

        else:
            # Default: stop
            if self.kinematics == "holonomic":
                return np.array([0.0, 0.0])
            else:
                return np.array([0.0, 0.0])
