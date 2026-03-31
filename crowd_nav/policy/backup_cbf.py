import numpy as np
import cvxopt
from numpy.linalg import norm
import logging
from typing import List

from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.state import ObservableState, FullState
from crowd_sim.envs.utils.human import Human

# Safety constraint (from DR-bCBF codebase)


class Constraint:
    def alpha(self, x):
        """
        Strengthening function.

        """
        return x
        # return 15 * x + x**3

    def alpha_b(self, x):
        """
        Strengthening function for reachability constraint.

        """
        return 10 * x

    def h1_x(self, x, humans: List[Human], human_radius, human_max_speed, backup_time):
        """
        Safety constraint.

        x: robot state
        humans: list of human states
        human_radius: human radius
        human_max_speed: human maximum speed
        backup_time: backup time

        Returns:
        horizontal_distance - reachability_distance

        At each propagated time step, check if the robot is within the reachability set of the human.

        """
        scores = []
        for human in humans:
            horizontal_distance = (x[0] - human.px) ** 2 + (x[1] - human.py) ** 2
            reachability_distance = (human_radius + human_max_speed * backup_time) ** 2
            scores.append(horizontal_distance - reachability_distance)
        return scores

    def grad_h1(self, x):
        """
        Gradient of safety constraint.

        """
        g = np.array([-1, 0])
        return g

    def hb_x(self, x):
        """
        Reachability constraint.

        """
        hb = -x[1]
        return hb

    def grad_hb(self, x):
        """
        Gradient of reachability constraint.

        """
        gb = np.array([0, -1])
        return gb


class ASIF(Constraint):
    def setupASIF(self, blending_bool=False, control_tightening=True) -> None:

        # Backup properties
        self.backupTime = 1.25  # [sec] (total backup time)
        self.backupTrajs = []
        self.backup_save_N = 5  # saves every N backup trajectory (for plotting)
        self.delta_array = [0]

        # Tightening constants
        self.Lh_const = 1
        self.Lhb_const = 1
        self.L_cl = 1  # Lipschitz constant of closed-loop dynamics

        # Blending constants
        self.blending_bool = blending_bool
        self.control_tightening = control_tightening

    def asif(self, x, u_des):
        """
        Implicit active set invariance filter (ASIF) using QP.

        """

        if self.blending_bool:
            # Introduce blending control method

            ###### Calculate implicit control barrier function

            # propogate flow under backup
            # Total backup trajectory time
            tmax_b = self.backupTime

            # Backup trajectory points
            rtapoints = int(math.ceil(tmax_b / self.del_t))

            # Discretization tightening constant
            mu_d = (self.del_t / 2) * self.Lh_const * (self.sup_fcl + self.dw_max)

            if len(self.delta_array) < rtapoints:
                for i in range(1, rtapoints):
                    # calculate tightening terms
                    t = self.del_t * i

                    # Gronwall bound
                    # delta_t = (self.dw_max / self.L_cl) * (np.exp(self.L_cl * t) - 1)

                    # Disturbance obs bound
                    e_bar = np.exp(-t) * self.dw_max + (self.dv_max) * (1 - np.exp(-t))
                    delta_t = ((self.dv_max / self.L_cl**2) + (e_bar / self.L_cl)) * (
                        np.exp(self.L_cl * t - 1) - (self.dv_max / self.L_cl) * t
                    )

                    # Tightening epsilon
                    epsilon = self.Lh_const * delta_t

                    self.delta_array.append(delta_t)

                    if i == rtapoints - 1:
                        # calculate epsilon_b
                        self.epsilon_b = self.Lhb_const * delta_t

            # State tracking array
            lenx = len(self.x0)
            phi = np.zeros((rtapoints, lenx))
            phi[0, :] = x

            # Simulate flow under backup control law
            new_x = x

            backupFlow = self.integrateStateBackup(
                new_x,
                np.arange(0, self.backupTime, self.del_t),
                self.int_options,
            )

            phi[:, :] = backupFlow[:, :].T

            # Store backup trajectories for plotting
            if self.curr_step % self.backup_save_N == 0:
                self.backupTrajs.append(phi)

            min_h_value = np.min(
                [
                    self.h1_x(phi[itx, :])
                    - int(self.control_tightening) * (self.delta_array[itx] + mu_d)
                    for itx in range(rtapoints)
                ]
            )
            hb_value = (
                self.hb_x(phi[-1, :]) - int(self.control_tightening) * self.epsilon_b
            )

            hi_x = np.min([min_h_value, hb_value])

            # Calculate blending function
            u_b = self.backupControl(x)

            u_act, lambda_score = self.blendInputs(x, u_des, u_b, np.max([hi_x, 0]))

            self.lambda_score = lambda_score

            # If safe action is different the desired action, RTA is intervening
            if np.linalg.norm(u_act - u_des) >= 0.0001:
                intervening = True
            else:
                intervening = False

            solver_dt = None

            return u_act, intervening, solver_dt

        # QP objective function
        M = np.eye(2)
        q = np.array(
            [u_des, 0.0]
        )  # Need to append the control with 0 to get at least 2 dimensions

        # Control constraints
        G = [[1.0, 0.0], [-1.0, 0.0]]
        h = [-self.u_max, -self.u_max]

        # Total backup trajectory time
        tmax_b = self.backupTime

        # Backup trajectory points
        rtapoints = int(math.ceil(tmax_b / self.del_t))

        # State tracking array
        lenx = len(self.x0)
        phi = np.zeros((rtapoints, lenx))
        phi[0, :] = x

        # Sensitivity matrix tracking array
        S = np.zeros((lenx, lenx, rtapoints))
        S[:, :, 0] = np.eye(lenx)

        # Simulate flow under backup control law
        new_x = np.concatenate((x, S[:, :, 0].flatten()))

        backupFlow = self.integrateStateBackup(
            new_x,
            np.arange(0, self.backupTime, self.del_t),
            self.int_options,
        )

        phi[:, :] = backupFlow[:lenx, :].T
        S[:, :, :] = backupFlow[lenx:, :].reshape(lenx, lenx, rtapoints)

        # Store backup trajectories for plotting
        if self.curr_step % self.backup_save_N == 0:
            self.backupTrajs.append(phi)

        fx_0 = self.f_x(x)
        gx_0 = self.g_x(x)

        # Construct barrier constraint for each point along trajectory
        for i in range(
            1, rtapoints
        ):  # Skip first point because of relative degree issue (general problem with BaCBFs)

            h_phi = self.h1_x(phi[i, :])
            gradh_phi = self.grad_h1(phi[i, :])
            g_temp_i = gradh_phi.T @ S[:, :, i] @ gx_0

            epsilon = 0
            robust_grad = 0
            if self.robust:
                t = self.del_t * i

                # Gronwall bound
                delta_t = (self.dw_max / self.L_cl) * (np.exp(self.L_cl * t) - 1)

                # Tightening epsilon
                epsilon = self.Lh_const * delta_t

                # Discretization tightening constant
                mu_d = (self.del_t / 2) * self.Lh_const * (self.sup_fcl + self.dw_max)

                # Robustness term
                robust_grad = np.linalg.norm(gradh_phi @ S[:, :, i]) * self.dw_max

                # Store only the first time
                if len(self.delta_array) < rtapoints:
                    self.delta_array.append(delta_t)
            else:
                # Discretization tightening constant
                mu_d = (self.del_t / 2) * self.Lh_const * self.sup_fcl

            h_temp_i = (
                -(gradh_phi @ S[:, :, i] @ fx_0 + self.alpha(h_phi - epsilon - mu_d))
                + robust_grad
            )

            # Append constraint
            G.append([g_temp_i, 0])
            h.append(h_temp_i)

            # Make sure last point is in the backup set
            if i == rtapoints - 1:

                hb_phi = self.hb_x(phi[i, :])
                gradhb_phi = self.grad_hb(phi[i, :])

                robust_grad_b = 0
                epsilonT = 0
                if self.robust:
                    # Tightening epsilon
                    epsilonT = self.Lhb_const * delta_t

                    # Robustness term
                    robust_grad_b = (
                        np.linalg.norm(gradhb_phi @ S[:, :, i]) * self.dw_max
                    )

                h_temp_i = (
                    -(gradhb_phi @ S[:, :, i] @ fx_0 + self.alpha_b(hb_phi - epsilonT))
                    + robust_grad_b
                )
                g_temp_i = gradhb_phi.T @ S[:, :, i] @ gx_0

                # Append constraint
                G.append([g_temp_i, 0])
                h.append(h_temp_i)

        # Solve QP
        try:
            tic = time.perf_counter()
            sltn = quadprog.solve_qp(M, q, np.array(G).T, np.array(h), 0)
            u_act = sltn[0]
            active_constraint = sltn[5]
            toc = time.perf_counter()
            solver_dt = toc - tic
            u_act = u_act[0]  # Only extract scalar we need
        except:
            u_act = -1
            solver_dt = None
            if self.verbose:
                print("no soltn")

        # If safe action is different the desired action, RTA is intervening
        if np.linalg.norm(u_act - u_des) >= 0.0001:
            intervening = True
        else:
            intervening = False

        return u_act, intervening, solver_dt


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
