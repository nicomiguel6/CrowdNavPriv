"""
--------------------------------------------------------------------------

van Wijk, David
Texas A&M University
Aerospace Engineering

Disturbance-Robust Backup Control Barrier Functions (DR-bCBF) code base.

© 2024 David van Wijk <davidvanwijk@tamu.edu>

---------------------------------------------------------------------------

Module contains plotting functions.

"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation
import numpy as np
import tqdm


class Plotter:
    def __init__(self):
        self.fig = None
        self.ax = None

    def plotter(
        self,
        x,
        u_act,
        intervening,
        u_p,
        env,
        lambda_scores,
        phase_plot_1=True,
        phase_plot_2a=True,
        phase_plot_2b=True,
        phase_plot_CI=False,
        phase_plot_nominal=False,
        control_plot=True,
        summary_plot=True,
        animate_pp1=True,
        latex_plots=False,
        save_plots=False,
        show_plots=True,
        legend_flag=False,
    ):
        if latex_plots:
            plt.rcParams.update(
                {
                    "text.usetex": True,
                    "font.family": "serif",
                }
            )

        alpha_set = 0.35
        title_sz, xaxis_sz, legend_sz, ticks_sz = 20, 23, 18, 16
        lwp = 2.2
        lwp_sets = 2
        x1 = x[0, :]
        x2 = x[1, :]

        def setupPhasePlot(fig=None, ax=None):
            colors = True
            # Plot limits
            x_max = 0.5
            x_min = -2
            y_max = 1.4
            y_min = -0.4
            x_c = np.linspace(-4, 0, 1000)
            y_c = np.sqrt(-2 * x_c * env.u_max)
            y_c_d = np.sqrt(-2 * x_c * (env.u_max - env.dw_max)) - env.dw_max

            # x1 vs x2 plot
            # plt.figure(figsize=(12.5, 8.5), dpi=100)
            fig, ax = plt.subplots(figsize=(12.5, 8.5), dpi=100)

            if legend_flag:
                plt.fill_between(
                    [0, x_max],
                    y_min,
                    y_max,
                    color=[249 / 255, 3 / 255, 255 / 255],  # Unsafe states are purple
                    alpha=alpha_set,
                    label="$\mathcal{X} \\backslash \mathcal{C}_{\\rm S}$ (Unsafe States)",
                )
                plt.vlines(
                    x=0.0,
                    ymin=0,
                    ymax=y_max,
                    color=[255 / 255, 0 / 255, 0 / 255],
                    linewidth=lwp_sets,
                )
                plt.hlines(
                    y=0.0,
                    xmin=x_min,
                    xmax=0,
                    color=[0 / 255, 176 / 255, 240 / 255],
                    linewidth=lwp_sets,
                    alpha=1,
                )
                if colors:
                    plt.fill_between(
                        x_c,
                        y_c_d,
                        y_max,
                        color=[
                            255 / 255,
                            0 / 255,
                            0 / 255,
                        ],  # State constraint sets are red
                        alpha=alpha_set,
                        label="$\mathcal{C}_{\\rm S} \\backslash \mathcal{C}_{\\rm I}$",
                    )
                    plt.vlines(
                        x=0.0,
                        ymin=y_min,
                        ymax=0,
                        color=[0 / 255, 176 / 255, 240 / 255],
                        alpha=1,
                        linewidth=lwp_sets,
                    )
                    plt.plot(
                        x_c,
                        y_c_d,
                        color=[255 / 255, 180 / 255, 123 / 255],
                        alpha=1,
                        linewidth=lwp_sets,
                    )
                    plt.fill_between(
                        x_c,
                        0,
                        y_min,
                        color=[193 / 255, 229 / 255, 245 / 255],
                        alpha=alpha_set,
                        label="$\mathcal{C}_{\\rm B}$",
                    )
                    if phase_plot_CI:
                        y_c_d = np.sqrt(-2 * x1 * (env.u_max - env.dw_max)) - env.dw_max
                        plt.fill_between(
                            x1,
                            x2,
                            y_c_d,
                            color=[
                                255 / 255,
                                180 / 255,
                                123 / 255,
                            ],  # Recoverable set CI is orange)
                            alpha=alpha_set,
                            label="$\mathcal{C}_{\\rm I}$",
                        )
                    else:
                        plt.fill_between(
                            x_c,
                            0,
                            # y_c,
                            y_c_d,
                            color=[255 / 255, 180 / 255, 123 / 255],
                            alpha=alpha_set,
                            label="$\mathcal{C}_{\\rm I}$",
                        )
            else:
                plt.fill_between(
                    [0, x_max],
                    y_min,
                    y_max,
                    color=[255 / 255, 204 / 255, 204 / 255],
                    alpha=alpha_set,
                )
                plt.vlines(
                    x=0.0,
                    ymin=0,
                    ymax=y_max,
                    color=[255 / 255, 0 / 255, 0 / 255],
                    linewidth=lwp_sets,
                )
                plt.hlines(
                    y=0.0,
                    xmin=x_min,
                    xmax=0,
                    color=[0 / 255, 176 / 255, 240 / 255],
                    linewidth=lwp_sets,
                    alpha=1,
                )
                if colors:
                    plt.fill_between(
                        x_c,
                        y_c_d,
                        y_max,
                        color=[255 / 255, 180 / 255, 123 / 255],
                        alpha=alpha_set,
                    )
                    plt.vlines(
                        x=0.0,
                        ymin=y_min,
                        ymax=0,
                        color=[0 / 255, 176 / 255, 240 / 255],
                        alpha=1,
                        linewidth=lwp_sets,
                    )
                    plt.plot(
                        x_c,
                        y_c_d,
                        color=[84 / 255, 130 / 255, 53 / 255],
                        alpha=1,
                        linewidth=lwp_sets,
                    )
                    plt.fill_between(
                        x_c,
                        0,
                        y_min,
                        color=[193 / 255, 229 / 255, 245 / 255],
                        alpha=alpha_set,
                    )
                    if phase_plot_CI:
                        y_c_d = np.sqrt(-2 * x1 * (env.u_max - env.dw_max)) - env.dw_max
                        plt.fill_between(
                            x1,
                            x2,
                            y_c_d,
                            color=[217 / 255, 242 / 255, 208 / 255],
                            alpha=alpha_set,
                        )
                    else:
                        plt.fill_between(
                            x_c,
                            0,
                            y_c_d,
                            color=[217 / 255, 242 / 255, 208 / 255],
                            alpha=alpha_set,
                        )

            plt.axis("equal")
            plt.xlabel(r"$x_1$", fontsize=xaxis_sz)
            plt.ylabel(r"$x_2$", fontsize=xaxis_sz)
            plt.xlim([x_min, x_max])
            plt.ylim([y_min, y_max])
            plt.xticks(fontsize=ticks_sz)
            plt.yticks(fontsize=ticks_sz)
            # plt.grid(True)
            # ax = plt.gca()
            ax.set_aspect("equal", adjustable="box")
            self.fig = fig
            self.ax = ax

        # Phase plot 1 (no disturbance radii)
        if phase_plot_1:
            setupPhasePlot()
            plt.plot(x1, x2, "-", color="magenta", linewidth=lwp, label="Trajectory")
            ax = plt.gca()

            if env.backupTrajs:
                for i, xy in enumerate(env.backupTrajs):
                    if i == 0:
                        label = "Nominal Backup Trajectory"
                    else:
                        label = None

                    plt.plot(
                        xy[:, 0],
                        xy[:, 1],
                        color="cyan",
                        linewidth=1.5,
                        label=label,
                        zorder=1,
                    )

            ax.legend(fontsize=legend_sz, loc="upper right")

        gw_edge_color = "#bababa"

        # Phase plot 2 (with GW disturbance radii)
        if phase_plot_2a:
            setupPhasePlot()
            plt.plot(x1, x2, "-", color="magenta", linewidth=lwp, label="Trajectory")

            ax = plt.gca()

            e_tstep = 10
            if env.backupTrajs:
                rta_points = len(env.backupTrajs[0])
                max_numBackup = len(env.backupTrajs)  # 15
                for i, xy in enumerate(env.backupTrajs):
                    if i == 0:
                        label = "Nominal Backup Trajectory"
                    else:
                        label = None

                    if i < max_numBackup:
                        circ = []
                        if env.robust:
                            for j in np.arange(0, rta_points, e_tstep):
                                t = j * env.del_t
                                r_t = env.delta_array[j]
                                cp = patches.Circle(
                                    (xy[j, 0], xy[j, 1]),
                                    r_t,
                                    color=gw_edge_color,
                                    fill=False,
                                    linestyle="--",
                                    label="GW Norm Ball",
                                )
                                if i == 0 and j == 0:
                                    ax.add_patch(cp)
                                circ.append(cp)
                            coll = PatchCollection(
                                circ,
                                zorder=100,
                                facecolors=("none",),
                                edgecolors=(gw_edge_color,),
                                linewidths=(1,),
                                linestyle=("--",),
                            )
                            ax.add_collection(coll)
                        plt.plot(
                            xy[:, 0],
                            xy[:, 1],
                            color="cyan",
                            linewidth=1.5,
                            label=label,
                            zorder=1,
                        )

            ax.legend(fontsize=legend_sz, loc="upper right")

        # Phase plot 2 (with GW disturbance radii)
        if phase_plot_2b:
            setupPhasePlot()
            plt.plot(x1, x2, "-", color="magenta", linewidth=lwp, label="Trajectory")

            ax = plt.gca()

            e_tstep = 1
            if env.backupTrajs:
                rta_points = len(env.backupTrajs[0])
                max_numBackup = len(env.backupTrajs)  # 15
                for i, xy in enumerate(env.backupTrajs):
                    if i == 0:
                        label = "Nominal Backup Trajectory"
                    else:
                        label = None

                    if i < max_numBackup:
                        if env.robust:
                            circ = []
                            for j in np.arange(0, rta_points, e_tstep):
                                t = j * env.del_t
                                r_t = env.delta_array[j]
                                if j == rta_points - 1:
                                    cp = patches.Circle(
                                        (xy[j, 0], xy[j, 1]),
                                        r_t,
                                        color=gw_edge_color,
                                        fill=False,
                                        linestyle="--",
                                        label="GW Norm Ball",
                                    )
                                    if i == 0:
                                        ax.add_patch(cp)
                                    circ.append(cp)
                            coll = PatchCollection(
                                circ,
                                zorder=100,
                                facecolors=("none",),
                                edgecolors=(gw_edge_color,),
                                linewidths=(1,),
                                linestyle=("--",),
                            )
                            ax.add_collection(coll)
                        plt.plot(
                            xy[:, 0],
                            xy[:, 1],
                            color="cyan",
                            linewidth=1.5,
                            label=label,
                            zorder=1,
                        )

            ax.legend(fontsize=legend_sz, loc="upper right")

        # Phase plot CI
        if phase_plot_CI:
            setupPhasePlot()

            booly = x2 >= 0
            plt.plot(
                x1[booly],
                x2[booly],
                color=[216 / 255, 110 / 255, 204 / 255],
                alpha=1,
                linewidth=lwp_sets,
            )
            if legend_flag:
                label = "$\mathcal{C}_{\\rm I}$"
            else:
                label = None
            plt.fill_between(
                x1[booly],
                0,
                # y_c,
                x2[booly],
                color=[242 / 255, 207 / 255, 238 / 255],
                alpha=alpha_set,
                label=label,
            )
            ax = plt.gca()

            lblsize = legend_sz * 1.35
            ax.text(-1.786, -0.25, "$\mathcal{C}_{\\rm B}$", fontsize=lblsize)
            ax.text(-1.786, 1.2, "$\mathcal{C}_{\\rm R}$", fontsize=lblsize)
            ax.text(
                0.17,
                1.2,
                "$\mathcal{X} \\backslash \mathcal{C}_{\\rm S}$",
                fontsize=lblsize,
            )

        delta_t = env.del_t
        t_span_u = np.arange(u_act.shape[1] - 1) * delta_t

        if phase_plot_nominal:
            setupPhasePlot()
            plt.plot(
                x[0, :],
                x[1, :],
                "-",
                color="black",
                linewidth=lwp,
                label="Desired Control Trajectory",
            )
            plt.title("Disturbed System following Nominal Control")
            ax = plt.gca()

            ax.legend(fontsize=legend_sz, loc="upper right")

        if control_plot:
            ax = plt.figure(figsize=(10, 7), dpi=100)
            ax = ax.add_subplot(111)
            ax.grid(True)
            plt.xticks(fontsize=ticks_sz)
            plt.yticks(fontsize=ticks_sz)
            color = "green"
            ax.plot(
                t_span_u,
                u_p[0][1:],
                "--",
                color="red",
                label="$u_{\\rm des}$",
                linewidth=lwp,
            )
            ax.plot(
                t_span_u,
                u_act[0][1:],
                "-",
                color=color,
                label="$u_{\\rm act}$",
                linewidth=lwp,
            )
            ax.plot(
                t_span_u,
                lambda_scores,
                "-",
                color="black",
                label="$\lambda$",
                linewidth=lwp,
            )
            ax.set_ylim(-1.2, 1.2)
            ax.set_ylabel("u", fontsize=xaxis_sz)

            ax.legend(fontsize=legend_sz, loc="lower right")
            plt.xlabel("time, t (s)", fontsize=xaxis_sz)
            if save_plots:
                plt.savefig("plots/control_plot.svg", dpi=100)

        if summary_plot:

            # Plot limits
            x_max = 0.5
            x_min = -2
            y_max = 1.4
            y_min = -0.4

            fig = plt.figure(figsize=(16, 9), dpi=1920 / 16)
            grid = plt.GridSpec(1, 2, hspace=0.1, wspace=0.4)

            traj_ax = fig.add_subplot(grid[0, 0])
            control_ax = fig.add_subplot(grid[0, 1])

            ## Format Axes

            traj_ax.set_xlim([x_min, x_max])
            traj_ax.set_ylim([y_min, y_max])
            title = "System Trajectory following Robust Blended Controller"
            traj_ax.set_title(title)
            traj_ax.set_xlabel(r"$x_1$ (position)")
            traj_ax.set_ylabel(r"$x_2$ (velocity)")

            control_ax.set_xlim([-0.2, 5.2])
            control_ax.set_ylim([-1.2, 1.2])
            title = "Control Input"
            control_ax.set_title(title)
            control_ax.set_xlabel(r"Time $t$")
            control_ax.set_ylabel(r"Control Input $u = \dot{x}_2$ (acceleration)")

            #
        if animate_pp1:
            n_frames = x1.shape[-1]
            progress_bar = tqdm.tqdm(total=n_frames)

            setupPhasePlot()

            # --- set up figure & axes once ---
            # ax.set_aspect("equal", "box")

            ax = self.ax
            fig = self.fig
            # main trajectory line
            (line_traj,) = ax.plot([], [], "k-", linewidth=lwp, label="Trajectory")

            # pre-create backup‐trajectory lines
            # backup_lines = []
            # for i in range(len(env.backupTrajs)):
            #     lbl = "Nominal Backup Trajectory" if i == 0 else None
            #     (ln,) = ax.plot(
            #         [], [], color="cyan", linewidth=1.5, label=lbl, zorder=1
            #     )
            #     backup_lines.append(ln)

            lbl = "Nominal Backup Trajectory"
            (backup_line,) = ax.plot(
                [], [], color="cyan", linewidth=1.5, label=lbl, zorder=1
            )

            circ = [
                patches.Circle(
                    (0, 0),  # dummy initial center
                    0,  # dummy radius
                    color=gw_edge_color,
                    fill=False,
                    linestyle="--",
                    label="GW Norm Ball",
                    zorder=100,
                    visible=False,
                )
                for jtx in range(len(env.backupTrajs[0]))
            ]

            ax.legend(fontsize=legend_sz, loc="upper right")

            def init():
                line_traj.set_data([], [])
                # for ln in backup_lines:
                #     ln.set_data([], [])
                backup_line.set_data([], [])
                for circle_patch in circ:
                    circle_patch.set_visible(False)
                    ax.add_patch(circle_patch)
                return [line_traj, backup_line, *circ]

            self.traj_counter = 0

            def update(n):
                # update main trajectory
                line_traj.set_data(x1[: n + 1], x2[: n + 1])
                progress_bar.update(1)

                rta_points = len(env.backupTrajs[0])
                max_numBackup = len(env.backupTrajs)  # 15

                # update backup trajectories [for n, check if its time to add in new backup (env.backup_save_N)]
                if n % env.backup_save_N == 0:

                    xy = env.backupTrajs[int(n / env.backup_save_N)]
                    # backup_lines[i].set_data(xy[:, 0], xy[:, 1])
                    backup_line.set_data(xy[:, 0], xy[:, 1])

                    # Circle Patch Collection
                    for j in np.arange(0, rta_points, e_tstep):
                        t = j * env.del_t
                        r_t = env.delta_array[j]
                        curr_patch = circ[j]
                        curr_patch.set_center((xy[j, 0], xy[j, 1]))
                        curr_patch.set_radius(r_t)
                        # curr_patch = patches.Circle(
                        #     (xy[j, 0], xy[j, 1]),
                        #     r_t,
                        #     color=gw_edge_color,
                        #     fill=False,
                        #     linestyle="--",
                        #     label="GW Norm Ball",
                        # )
                        curr_patch.set_visible(True)
                        # if i == 0 and j == 0:
                        #     ax.add_patch(curr_patch)

                    self.traj_counter += 1

                # close progress bar on last frame
                if n == n_frames - 1:
                    progress_bar.close()

                return [line_traj, backup_line, *circ]

            ani = animation.FuncAnimation(
                fig,
                update,
                frames=n_frames,
                init_func=init,
                blit=True,
                interval=1,
            )

            writervideo = animation.FFMpegWriter(fps=10)
            ani.save("animation.mp4", writer=writervideo)
            plt.close(fig)

        gw_edge_color = "#bababa"

        if show_plots:
            plt.show()
