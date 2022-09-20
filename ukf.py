import numpy as np
import scipy.linalg
from copy import deepcopy
from threading import Lock


class UKFException(Exception):
    """Raise for errors in the UKF, usually due to bad inputs"""


class UKF:
    def __init__(self, num_states, process_noise, initial_state, initial_covar, alpha, k, beta, iterate_function):
        """
        Initializes the unscented kalman filter
        :param num_states: int, the size of the state
        :param process_noise: the process noise covariance per unit time, should be num_states x num_states
        :param initial_state: initial values for the states, should be num_states x 1
        :param initial_covar: initial covariance matrix, should be num_states x num_states, typically large and diagonal
        :param alpha: UKF tuning parameter, determines spread of sigma points, typically a small positive value
        :param k: UKF tuning parameter, typically 0 or 3 - num_states
        :param beta: UKF tuning parameter, beta = 2 is ideal for gaussian distributions
        :param iterate_function: function that predicts the next state
                    takes in a num_states x 1 state and a float timestep
                    returns a num_states x 1 state
        """
        self.n_dim = int(num_states)
        self.n_sig = 1 + num_states * 2
        self.q = process_noise
        self.x = initial_state
        self.p = initial_covar
        self.beta = beta
        self.alpha = alpha
        self.k = k
        self.iterate = iterate_function
        self.states = np.array([
            [0,0,0,0,1,0],
            [0,0,0,0,0,1],
            [0,0,1,0,0,0],
            [0,0,0,1,0,0]
        ])
        
        self.lambd = pow(self.alpha, 2) * (self.n_dim + self.k) - self.n_dim

        self.covar_weights = np.zeros(self.n_sig)
        self.mean_weights = np.zeros(self.n_sig)

        self.covar_weights[0] = (self.lambd / (self.n_dim + self.lambd)) + (1 - pow(self.alpha, 2) + self.beta)
        self.mean_weights[0] = (self.lambd / (self.n_dim + self.lambd))

        for i in range(1, self.n_sig):
            self.covar_weights[i] = 1 / (2*(self.n_dim + self.lambd))
            self.mean_weights[i] = 1 / (2*(self.n_dim + self.lambd))

        self.sigmas = self.__get_sigmas()

        self.lock = Lock()

    def __get_sigmas(self):
        """generates sigma points"""
        ret = np.zeros((self.n_dim , self.n_sig))

        tmp_mat = (self.n_dim + self.lambd)*self.p
        spr_mat = scipy.linalg.sqrtm(tmp_mat)
        
        ret[:,0:1] = self.x
        for i in range(self.n_dim):
            ret[:,i+1:i+2] = self.x + spr_mat[:,i:i+1] 
            ret[:,i+1+self.n_dim:i+2+self.n_dim] = self.x - spr_mat[:,i:i+1]

        return ret

    def update(self, y_actual, r_matrix):
        """
        performs a measurement update
        :param states: list of indices (zero-indexed) of which states were measured, that is, which are being updated
        :param data: list of the data corresponding to the values in states
        :param r_matrix: error matrix for the data, again corresponding to the values in states
        """
        self.lock.acquire()

        num_states = len(self.states)
        
        # create y, sigmas of just the states that are being updated
        y = self.states @ self.sigmas 
        # create y_mean, the mean of just the states that are being updated
        y_mean = self.states @ self.x
        # differences in y from y mean
        y_diff = y - y_mean
        x_diff = self.sigmas - self.x
        # covariance of measurement
        p_yy = np.zeros((num_states, num_states))
        for i, val in enumerate(np.array_split(y_diff, self.n_sig, 1)):
            p_yy += self.covar_weights[i] * val @ val.T
        # add measurement noise
        p_yy += r_matrix

        # covariance of measurement with states
        p_xy = np.zeros((self.n_dim, num_states))
        for i, val in enumerate(zip(np.array_split(y_diff, self.n_sig, 1), np.array_split(x_diff, self.n_sig, 1))):
            p_xy += self.covar_weights[i] * val[1] @ val[0].T

        k = p_xy @ np.linalg.inv(p_yy)
        self.x += k @ (y_actual - y_mean)
        self.p -= k @ (p_yy @ k.T) 
        self.sigmas = self.__get_sigmas()

        self.lock.release()

    def predict(self, timestep):
        """
        performs a prediction step
        :param timestep: float, amount of time since last prediction
        """

        self.lock.acquire()
        
        sigmas_out = self.iterate(self.sigmas,timestep)
        x_out = (sigmas_out @ self.mean_weights.reshape(-1,1))
        
        p_out = np.zeros((self.n_dim, self.n_dim))
        # for each sigma point
        for i in range(self.n_sig):
            # take the distance from the mean
            # make it a covariance by multiplying by the transpose
            # weight it using the calculated weighting factor
            # and sum
            diff = sigmas_out[:,i:i+1] - x_out
            p_out += self.covar_weights[i] * (diff @ diff.T)
        # add process noise
        p_out += timestep * self.q

        self.sigmas = sigmas_out
        self.x = x_out
        self.p = p_out

        self.lock.release()

    def get_state(self, index=-1):
        """
        returns the current state (n_dim x 1), or a particular state variable (float)
        :param index: optional, if provided, the index of the returned variable
        :return:
        """
        if index >= 0:
            return self.x[index]
        else:
            return self.x

    def get_covar(self):
        """
        :return: current state covariance (n_dim x n_dim)
        """
        return self.p

    def set_state(self, value, index=-1):
        """
        Overrides the filter by setting one variable of the state or the whole state
        :param value: the value to put into the state (1 x 1 or n_dim x 1)
        :param index: the index at which to override the state (-1 for whole state)
        """
        with self.lock:
            if index != -1:
                self.x[index] = value
            else:
                self.x = value

    def reset(self, state, covar):
        """
        Restarts the UKF at the given state and covariance
        :param state: n_dim x 1
        :param covar: n_dim x n_dim
        """

        with self.lock:
            self.x = state
            self.p = covar