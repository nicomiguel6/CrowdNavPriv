import numpy as np
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.action import ActionAcceleration


class Linear(Policy):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.kinematics = "holonomic"
        self.multiagent_training = True

    def configure(self, config):
        assert True

    def predict(self, state):
        self_state = state.self_state
        theta = np.arctan2(self_state.gy - self_state.py, self_state.gx - self_state.px)
        vx = np.cos(theta) * self_state.v_pref
        vy = np.sin(theta) * self_state.v_pref
        action = ActionXY(vx, vy)

        return action


class LinearAcceleration(Policy):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.kinematics = "holonomic"
        self.multiagent_training = True

    def configure(self, config):
        assert True

    def predict(self, state):
        self_state = state.self_state
        theta = np.arctan2(self_state.gy - self_state.py, self_state.gx - self_state.px)
        vx = np.cos(theta) * self_state.v_pref
        vy = np.sin(theta) * self_state.v_pref

        # if at goal, set velocity to 0
        if (
            np.linalg.norm(
                np.array([self_state.gx - self_state.px, self_state.gy - self_state.py])
            )
            < 0.1
        ):
            vx = 0
            vy = 0

        # Compute the acceleration needed to reach the desired velocity
        ax = (vx - self_state.vx) / self.time_step
        ay = (vy - self_state.vy) / self.time_step

        return ActionAcceleration(ax, ay)


class GoStraight(Policy):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.kinematics = "holonomic"
        self.multiagent_training = True

    def configure(self, config):
        assert True

    def predict(self, state):
        self_state = state.self_state
        vx = self_state.v_pref
        vy = 0

        ax = (vx - self_state.vx) / self.time_step

        return ActionAcceleration(ax, 0)
