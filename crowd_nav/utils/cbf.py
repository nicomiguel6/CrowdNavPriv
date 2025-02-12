import numpy as np
import torch
import cvxopt

'''
Class to implement ControlBarrierFunction between robot and obstacle.

At each time step, high-level policy action is checked against specific obstacle.
CBF kicks in if action will result in unsafe set violation.
'''


class ControlBarrierFunction(object):
    def __init__(self, human, cbf_rate, state_dim, action_dim):
        '''
        NOTE THAT ALL FUNCTIONS MUST RETURN NP.ARRAYS
        '''
        self.human = human  # identifies which specific obstacle CBF is tracking
        self.cbf_rate = cbf_rate  # gamma value corresponding to CBF rate
        self.state_dim = state_dim # dimension of state, in unicyce case should be [px, py, theta] (holonomic)
        self.action_dim = action_dim # dimension of action, in unicycle case should be [v, w]
        '''
        xdot = f(x) + g(x)*u
        '''
        self._dynamics_drift_function = None # f(x)
        self._dynamics_control_function = None # g(x)

        '''
        h_dot = dh/dx * dx/dt = dh/dx (f(x) + g(x)*u) = Lfh + Lgh*u
        '''
        self._control_barrier_function = None # h(x, t)
        self._control_barrier_function_dx = None # dh/dt (x, t)

        #self._lie_derivative_f = None # Lfh
        #self._lie_derivative_g = None # Lgh

        '''
        Telemetry
        '''
        self.hs = []

    def quadratic_program_safety_filter(self, robot_state, human_state, reference_action):
        '''
        Filter control input for safety using CBF-QP
        '''

        
        # Calculate f(x)
        f = self.dynamics_drift_function(robot_state, human_state)

        # Calculate g(x)
        g = self.dynamics_control_function(robot_state, human_state)

        # Calculate h(x)
        h = self.control_barrier_function(robot_state, human_state)

        # Calculate dh/dx
        dhdx = self.control_barrier_function_dx(robot_state, human_state, reference_action)

        # Calculate Lie derivatives
        Lfh = dhdx @ f
        Lgh = dhdx @ g

        # Calculate psi
        psi = Lfh + Lgh@reference_action + self.cbf_rate*h

        ## Set up QP

        G = cvxopt.matrix([-Lgh])
        h_qp = cvxopt.matrix([Lfh + self.cbf_rate*h])

        P = cvxopt.matrix([np.eye(self.state_dim)])
        q = cvxopt.matrix([-reference_action])

        ## Solve QP

        filtered_action = cvxopt.solvers.qp(P, q, G, h_qp)

        filtered_action = filtered_action['x'][:]

        return np.array(filtered_action)

        

    @property
    def dynamics_drift_function(self): 
        return self._dynamics_drift_function
    
    @dynamics_drift_function.setter 
    def dynamics_drift_function(self, dynamic_drift_method):
        self._dynamics_drift_function = dynamic_drift_method

    @property
    def dynamics_control_function(self):
        return self._dynamics_control_function
    
    @dynamics_control_function.setter
    def dynamics_control_function(self, dynamic_control_method):
        self._dynamics_control_function = dynamic_control_method

    @property
    def control_barrier_function(self):
        return self._control_barrier_function

    @control_barrier_function.setter
    def control_barrier_function(self, control_barrier_method):
        self._control_barrier_function = control_barrier_method

    @property
    def control_barrier_function_dx(self):
        return self._control_barrier_function_dx

    @control_barrier_function_dx.setter
    def control_barrier_function_dx(self, control_barrier_function_dx):
        self.control_barrier_function_dx = control_barrier_function_dx

    # @property
    # def lie_derivative_f(self):
    #     return self._lie_derivative_f

    # @lie_derivative_f.setter
    # def lie_derivative_f(self, lie_derivative_f):
    #     self.lie_derivative_f = lie_derivative_f

    # @property
    # def lie_derivative_g(self):
    #     return self._lie_derivative_g

    # @lie_derivative_g.setter
    # def lie_derivative_g(self, lie_derivative_g):
    #     self.lie_derivative_g = lie_derivative_g
