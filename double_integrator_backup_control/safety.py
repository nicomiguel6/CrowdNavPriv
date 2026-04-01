"""
--------------------------------------------------------------------------

van Wijk, David
Texas A&M University
Aerospace Engineering

Disturbance-Robust Backup Control Barrier Functions (DR-bCBF) code base.

© 2024 David van Wijk <davidvanwijk@tamu.edu>

---------------------------------------------------------------------------

Module contains functions required for safety-critical control using control barrier functions.

"""

import numpy as np
import time
import math
import quadprog
from scipy.integrate import quad


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

    def h1_x(self, x):
        """
        Safety constraint.

        """
        h = -x[0]
        return h

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

            u_act, lambda_score = self.blendInputs(x, u_des, u_b, np.max(hi_x, 0))

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
