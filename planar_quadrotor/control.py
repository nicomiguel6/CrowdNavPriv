"""
--------------------------------------------------------------------------

van Wijk, David
Texas A&M University
Aerospace Engineering

Disturbance-Robust Backup Control Barrier Functions (DR-bCBF) code base.

© 2024 David van Wijk <davidvanwijk@tamu.edu>

---------------------------------------------------------------------------

Module containing control laws.

"""

import numpy as np


class Control:
    def setupControl(
        self,
    ) -> None:

        # Control limits
        self.F_max = 20
        self.M_max = 20

        self.F_bounds = [0.0, self.F_max]
        self.M_bounds = [-self.M_max, self.M_max]
        self.u_bounds = [[0.0, -self.M_max], [self.F_max, self.M_max]]

        self.Kp = 1.0
        self.Kd = 1.01
        self.J = 0.25
        self.m = 1
        self.beta = 1000.0

        self.kappa = 5
        self.theta_max = 55
        self.thetadot_max = 3

    def primaryControl(self, x_curr, t):
        """
        Primary controller producing desired control at each step.

        """
        u_des = np.array([0.0, 0.0])
        return u_des

    def backupControl(self, x):
        """
        Safe backup controller. Constant for this application.

        """
        u_b = np.array(
            [
                self.F_max,
                np.clip(self.Kp * x[2] + self.Kd * x[5], -self.M_max, self.M_max),
            ]
        )
        return u_b

    def lambdaScore(self, x, hi_min):

        # assume flow using backup controller has been propagated already

        # assume array of h and hb evaluated across flow has been evaluated

        # assume self.hi_max has been calculated already for state x

        lambda_score = 1 - np.exp(-self.beta * hi_min)  # / np.max([x[1], 0.0001]))

        return lambda_score

    def blendInputs(self, x, u_des, u_backup, hi_min):

        # Calculate lambda_score
        lambda_score = self.lambdaScore(x, hi_min)

        return (
            np.clip(
                lambda_score * u_des + (1 - lambda_score) * u_backup,
                self.u_bounds[0],
                self.u_bounds[1],
            ),
            lambda_score,
        )
