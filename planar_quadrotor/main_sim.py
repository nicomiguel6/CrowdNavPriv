"""
--------------------------------------------------------------------------

van Wijk, David
Texas A&M University
Aerospace Engineering

Disturbance-Robust Backup Control Barrier Functions (DR-bCBF) code base.

© 2024 David van Wijk <davidvanwijk@tamu.edu>

---------------------------------------------------------------------------

Module runs full simulation for double integrator example.

"""

import json
from datetime import datetime
import numpy as np
from safety import ASIF
from control import Control
from dynamics import Dynamics
from plotting import Plotter


class Simulation(ASIF, Control, Dynamics):
    def __init__(
        self,
        safety_flag=True,
        verbose=True,
        robust=False,
        dw_bool=False,
        blending_bool=False,
        primary_control=False,
        control_tightening=True,
        observer=True,
    ) -> None:

        self.blending_bool = blending_bool
        self.primary_control = primary_control
        self.control_tightening = control_tightening
        self.observer = observer

        self.setupControl()
        self.setupDynamics(self.blending_bool)
        self.setupASIF(self.blending_bool, self.control_tightening)

        self.verbose = verbose
        self.safety_flag = safety_flag
        self.robust = robust
        self.dw_bool = dw_bool

    def checkViolation(self, x_curr):
        """
        Check for safety violation.

        """
        h_funs = [lambda x: self.h1_x(x)]
        for i in range(len(h_funs)):
            h_x = h_funs[i]
            if h_x(x_curr) < 0:
                if self.verbose:
                    print(f"Safety violation, constraint {i+1}")

    def sim(self):
        """
        Simulates trajectory for pre-specified number of timesteps and performs
        point-wise safety-critical control modifications if applicable.

        """
        x0 = self.x0
        total_steps = self.total_steps

        # Tracking variables
        x_full = np.zeros((len(x0), total_steps))
        u_act_full = np.zeros((len(self.u_bounds), total_steps))
        u_des_full = np.zeros((len(self.u_bounds), total_steps))
        lambda_scores = []
        solver_times, avg_solver_t, max_solver_t, intervened = [], [], [], []
        x_full[:, 0] = x0
        x_curr = x0

        # Disturbance variables
        self.dw_full = np.zeros((len(x0), total_steps))

        # Observer variables
        xi_full = np.zeros((len(x0), total_steps))  # auxiliary state at each time step
        d_hat_full = np.zeros(
            (len(x0), total_steps)
        )  # disturbance estimate at each time step
        self.disturbance_gain = 10
        self.disturbance_gain_matrix = self.disturbance_gain * np.eye(len(x0))

        xi_full[:, 0] = x0
        xi_curr = xi_full[:, 0]
        d_hat_full[:, 0] = np.zeros_like(x0)
        d_hat_curr = d_hat_full[:, 0]
        self.d_hat_curr = d_hat_curr

        # Main loop
        for i in range(1, total_steps):
            t = self.curr_step * self.del_t

            # Generate desired control
            u_des = self.primaryControl(x_curr, t)
            u_des_full[:, i] = u_des

            # If safety check on, monitor control using active set invariance filter with DR-bCBF
            if self.safety_flag and not self.primary_control:
                u, boolean, sdt = self.asif(x_curr, u_des, t)
                solver_times.append(sdt)
                if boolean:
                    intervened.append(i)
                if hasattr(self, "lambda_score"):
                    lambda_scores.append(self.lambda_score)
            else:
                u = u_des

            # Compute disturbance vector (if applicable)
            if self.dw_bool:
                dw = self.disturbanceFun(t, x_curr, u, {})
                self.dw_full[:, i] = dw
            else:
                dw = np.zeros(len(x0))

            u_act_full[:, i] = u

            # Propagate auxiliary state with control
            xi_curr = self.integrateState(
                x_curr,
                u,
                self.del_t,
                d_hat_curr,
                self.int_options,
            )

            # Propagate states with control and disturbances (if applicable)
            x_curr = self.integrateState(
                x_curr,
                u,
                self.del_t,
                dw,
                self.int_options,
            )

            # Propagate estimated disturbance
            d_hat_curr = self.disturbance_gain_matrix @ (x_curr - xi_curr)

            # dx = self.f_x(x_curr) + self.g_x(x_curr) @ u
            # x_curr += dx * self.del_t
            x_full[:, i] = x_curr
            xi_full[:, i] = xi_curr
            d_hat_full[:, i] = d_hat_curr

            self.curr_step += 1
            self.d_hat_curr = d_hat_curr
            print("Current disturbance estimation: " + str(self.d_hat_curr))
            # if self.h1_x(x_curr) < 0:
            #     print("Crashed")

        self.xi_full = xi_full
        self.d_hat_full = d_hat_full

        if self.verbose and self.safety_flag:
            solver_times = [
                i for i in solver_times if i is not None
            ]  # Remove Nones which can be due to QP failures
            if solver_times:
                avg_solver_t = 1000 * np.average(solver_times)  # in ms
                max_solver_t = 1000 * np.max(solver_times)  # in ms
                print(f"Average solver time: {avg_solver_t:0.4f} ms")
                print(f"Maximum single solver time: {max_solver_t:0.4f} ms")

        return (
            x_full,
            total_steps,
            u_des_full,
            u_act_full,
            intervened,
            avg_solver_t,
            max_solver_t,
            lambda_scores,
        )


if __name__ == "__main__":

    # deltad = [0.05, 0.1, 0.25, 0.5]  # Disturbance magnitudes
    deltad = [np.sqrt(1.5)]  # For testing purposes with observer

    x_all = []
    u_des_all = []
    u_act_all = []
    lambda_scores_all = []
    env_all = []

    for delta_d in deltad:

        env = Simulation(
            safety_flag=True,
            verbose=True,
            robust=True,
            dw_bool=True,
            blending_bool=True,
            primary_control=False,
            control_tightening=True,
        )
        env.dw_max = delta_d
        print(
            "Running simulation with parameters:",
            "Safety:",
            env.safety_flag,
            "| Robustness:",
            env.robust,
            "| Process dist:",
            env.dw_bool,
            "| dw_max:",
            env.dw_max,
        )

        (
            x_full,
            total_steps,
            u_des_full,
            u_act_full,
            intervened,
            avg_solver_t,
            max_solver_t,
            lambda_scores,
        ) = env.sim()

        filename = (
            "data\\Disturbance_deltad_{}".format(
                env.dw_max
            )  # for experiment without observers
            if not hasattr(env, "dv_max")
            else "data\\Disturbance_deltad_{}_deltav_{}".format(env.dw_max, env.dv_max)
        )

        data_store_dict = {
            "x_full": x_full,
            "total_steps": total_steps,
            "u_des_full": u_des_full,
            "u_act_full": u_act_full,
            "intervened": intervened,
            "avg_solver_t": avg_solver_t,
            "max_solver_t": max_solver_t,
            "lambda_scores": lambda_scores,
        }

        attrs = {k: v for k, v in env.__dict__.items() if not k.startswith("_")}

        data_store_dict.update(attrs)

        # np.savez(filename + ".npz", **data_store_dict)

        # with open(filename, "w") as outfile:
        #     json.dump(data_store_dict, outfile)

        x_all.append(x_full)
        u_des_all.append(u_des_full)
        u_act_all.append(u_act_full)
        lambda_scores_all.append(lambda_scores)
        env_all.append(env)

    p = Plotter()

    p.plotter(
        x_all,
        u_act_all,
        intervened,
        u_des_all,
        env_all,
        lambda_scores_all,
        phase_plot_1=False,
        phase_plot_2a=True,
        phase_plot_2b=False,
        phase_plot_CI=False,
        phase_plot_nominal=False,
        summary_plot=False,
        control_plot=False,
        lambda_plot=False,
        animate_pp1=False,
        latex_plots=True,
        save_plots=False,
        show_plots=True,
        legend_flag=True,
        all=True,
    )
