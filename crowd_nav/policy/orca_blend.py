import logging
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.action import ActionXY


class OrcaBlend(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'ORCA-Blend'
        self.trainable = False
        self.multiagent_training = None
        self.kinematics = 'holonomic'
        self.reference_policy = None
        self.orca_policy = ORCA()
        self.blend_weight = 0.5

    def configure(self, config):
        if config.has_section('orca_blend'):
            self.blend_weight = config.getfloat('orca_blend', 'blend_weight')

    def set_reference_policy(self, policy):
        self.reference_policy = policy
        if policy is not None:
            self.kinematics = policy.kinematics

    def set_phase(self, phase):
        self.phase = phase
        if self.reference_policy is not None:
            self.reference_policy.set_phase(phase)
        self.orca_policy.set_phase(phase)

    def set_device(self, device):
        self.device = device
        if self.reference_policy is not None:
            self.reference_policy.set_device(device)
        self.orca_policy.set_device(device)

    def set_env(self, env):
        self.env = env
        if self.reference_policy is not None:
            self.reference_policy.set_env(env)
        self.orca_policy.set_env(env)

    def _sync_time_step(self):
        if self.reference_policy is not None:
            self.reference_policy.time_step = self.time_step
        self.orca_policy.time_step = self.time_step

    def predict(self, state):
        if self.reference_policy is None:
            raise AttributeError('Reference policy must be set before calling predict')
        if self.kinematics != 'holonomic':
            raise ValueError('ORCA blend only supports holonomic kinematics')
        if not 0.0 <= self.blend_weight <= 1.0:
            logging.warning('Blend weight out of [0, 1]: %s', self.blend_weight)

        self._sync_time_step()
        reference_action = self.reference_policy.predict(state)
        orca_action = self.orca_policy.predict(state)

        if not isinstance(reference_action, ActionXY) or not isinstance(orca_action, ActionXY):
            raise ValueError('ORCA blend expects ActionXY from both policies')

        weight = self.blend_weight
        blended_vx = (1 - weight) * reference_action.vx + weight * orca_action.vx
        blended_vy = (1 - weight) * reference_action.vy + weight * orca_action.vy
        return ActionXY(blended_vx, blended_vy)
