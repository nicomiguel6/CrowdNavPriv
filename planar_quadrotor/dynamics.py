"""
--------------------------------------------------------------------------

van Wijk, David
Texas A&M University
Aerospace Engineering

Disturbance-Robust Backup Control Barrier Functions (DR-bCBF) code base.

© 2024 David van Wijk <davidvanwijk@tamu.edu>

---------------------------------------------------------------------------

Module contains functions required for propagating dynamics of double integrator.

"""

import numpy as np
from scipy.integrate import solve_ivp
import jax.numpy as jnp


class Dynamics:
    def setupDynamics(
        self,
        blending_bool=False,
    ) -> None:

        # Integration options
        self.int_options = {"rtol": 1e-3, "atol": 1e-3}
        self.blending_bool = blending_bool

        # Simulation data
        self.del_t = 0.02  # [sec]
        self.tf = 3.0
        self.total_steps = int(self.tf / self.del_t) + 2
        self.curr_step = 0

        # Initial conditions
        self.x0 = np.array([0.0, 5.0, 10.0, 0.0, 0.0, 0.0])

        # Disturbances
        self.dw_max = np.sqrt(1.5)
        # self.dw_max = 0.08
        self.omega = 0.3
        self.dv_max = self.omega * self.dw_max

        # Constant
        max_xvel = np.sqrt(
            2 * (self.tf * self.F_max * np.sin(np.radians(self.theta_max)) / self.m)
            + 2 * self.tf
        )
        max_zvel = np.sqrt(2 * self.tf * (self.F_max / self.m - 9.81 + 0.5))
        self.sup_fcl = np.sqrt(
            max_xvel**2
            + max_zvel**2
            + self.thetadot_max**2
            + (self.F_max * np.sin(np.radians(self.theta_max)) / self.m + 1) ** 2
            + (self.F_max / self.m + 0.5 - 9.81) ** 2
            + (self.M_max / self.J) ** 2
        )

    def propMain(self, t, x, u, dist, args):
        """
        Propagation function for dynamics with disturbance and STM if applicable.
        Could be optimized (linear system).

        """
        lenx = len(self.x0)
        dx = np.zeros_like(x)
        dx[:lenx] = self.f_x(x[:lenx]) + self.g_x(x[:lenx]) @ u + dist
        if len(x) > lenx:
            # Construct F
            F = self.computeJacobianSTM(x[:lenx])

            # Extract STM & reshape
            STM = x[lenx:].reshape(lenx, lenx)
            dSTM = F @ STM

            # Reshape back to column
            dSTM = dSTM.reshape(lenx**2)
            dx[lenx:] = dSTM

        return dx

    def computeJacobianSTM(self, x):
        """
        Compute Jacobian of dynamics.
        """
        jac = self.A
        return jac

    def f_x(self, x):
        """
        Function f(x) for control affine dynamics, x_dot = f(x) + g(x)u.

        """
        f = np.array([*x[-3:], 0.0, -9.81, 0.0])
        return f

    def g_x(self, x):
        """
        Function g(x) for control affine dynamics, x_dot = f(x) + g(x)u.

        """
        g = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [np.sin(np.radians(x[2])), 0.0],
                [np.cos(np.radians(x[2])), 0.0],
                [0.0, -1 / 0.25],
            ]
        )
        return g

    def integrateState(self, x, u, t_step, dist, options):
        """
        State integrator using propagation function.

        """
        t_step = (0.0, t_step)
        args = {}

        soltn = solve_ivp(
            lambda t, x: self.propMain(t, x, u, dist, args),
            t_step,
            x,
            method="RK45",
            rtol=options["rtol"],
            atol=options["atol"],
        )
        x = soltn.y[:, -1]  # * self.s
        return x

    def propMainBackup(self, t, x, args):
        """
        Propagation function for backup dynamics and STM if applicable.

        """
        lenx = len(self.x0)
        dx = np.zeros_like(x)
        dx[:lenx] = self.f(x[:lenx]) + self.g(x[:lenx]) * self.backupControl(x[:lenx])

        # Construct F
        F = self.f(x[:lenx])

        # Extract STM & reshape
        STM = x[lenx:].reshape(lenx, lenx)
        dSTM = F @ STM

        # Reshape back to column
        dSTM = dSTM.reshape(lenx**2)
        dx[lenx:] = dSTM

        return dx

    def propMainBackupBlending(self, t, x, args):
        """Propagation function for nominal backup dynamics

        Args:
            t (_type_): _description_
            x (_type_): _description_
            args (_type_): _description_
        """
        lenx = len(self.x0)
        dx = np.zeros_like(x)
        if self.observer:
            dx = self.f_x(x) + self.g_x(x) @ self.backupControl(x) + self.d_hat_curr
        else:
            dx = self.f_x(x) + self.g_x(x) @ self.backupControl(x)

        return dx

    def integrateStateBackup(self, x, tspan_b, options):
        """
        Propagate backup flow over the backup horizon. Evaluate at discrete points.

        """
        t_step = (0.0, tspan_b[-1])
        args = {}
        if self.blending_bool is False:
            soltn = solve_ivp(
                lambda t, x: self.propMainBackup(t, x, args),
                t_step,
                x,
                method="RK45",
                rtol=options["rtol"],
                atol=options["atol"],
                t_eval=tspan_b,
            )
        else:

            soltn = solve_ivp(
                lambda t, x: self.propMainBackupBlending(t, x, args),
                t_step,
                x,
                method="RK45",
                rtol=options["rtol"],
                atol=options["atol"],
                t_eval=tspan_b,
            )
        x = soltn.y[:, :]
        # return (x[:,].T * self.s).T
        return x

    def integrateStateBackupwithDhat(self, x, tspan_b, options):
        """
        Propagate backup flow over the backup horizon . Evaluate at discrete points.

        """
        t_step = (0.0, tspan_b[-1])
        args = {}
        if self.blending_bool is False:
            soltn = solve_ivp(
                lambda t, x: self.propMainBackup(t, x, args),
                t_step,
                x,
                method="RK45",
                rtol=options["rtol"],
                atol=options["atol"],
                t_eval=tspan_b,
            )
        else:

            soltn = solve_ivp(
                lambda t, x: self.propMainBackupBlending(t, x, args),
                t_step,
                x,
                method="RK45",
                rtol=options["rtol"],
                atol=options["atol"],
                t_eval=tspan_b,
            )
        x = soltn.y[:, :]
        # return (x[:,].T * self.s).T
        return x

    def disturbanceFun(self, t, x, u, args):
        """
        Process disturbance function, norm bounded by dw_max.

        """
        dist_t = np.array([0, 0, 0, 1, 0.5 * np.sin(self.omega * t - (np.pi / 3)), 0])
        # dist_t = np.random.uniform(-np.ones((1, 6)), np.ones((1, 6)))
        dist_t = dist_t / np.linalg.norm(dist_t)
        # dist_t = np.array(
        #     [np.sin(self.omega * t + np.pi / 4), np.cos(self.omega * t + np.pi / 4)]
        # )
        return self.dw_max * dist_t
