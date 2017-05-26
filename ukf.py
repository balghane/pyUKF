import numpy as np
import scipy.linalg
from copy import deepcopy


class UKF:
    def __init__(self, num_states, process_noise, initial_state, initial_covar, alpha, k, beta, iterate_function):
        """
        Initializes the unscented kalman filter
        :param num_states: int, the size of the state
        :param process_noise: the process noise covariance, should be num_states x num_states
        :param initial_state: initial values for the states, should be num_states x 1
        :param initial_covar: initial covariance matrix, should be num_states x num_states, typically large and diagonal
        :param alpha: UKF tuning parameter, determines spread of sigma points, typically a small positive value
        :param k: UKF tuning parameter, typically 0 or 3 - num_states
        :param beta: UKF tuning parameter, beta = 2 is ideal for gaussian distributions
        :param iterate_function: function that predicts the next state
                    takes in a num_states x 1 state and a float timestep
                    returns a num_states x 1 state
        """
        self.n_dim = num_states
        self.n_sig = 1 + num_states * 2
        self.q = process_noise
        self.x = initial_state
        self.p = initial_covar
        self.beta = beta
        self.alpha = alpha
        self.k = k
        self.iterate = iterate_function

        self.lambd = pow(self.alpha, 2) * (self.n_dim + self.k) - self.n_dim

        self.covar_weights = np.zeros(self.n_sig)
        self.mean_weights = np.zeros(self.n_sig)

        self.covar_weights[0] = (self.lambd / (self.n_dim + self.lambd)) + (1 - pow(self.alpha, 2) + self.beta)
        self.mean_weights[0] = (self.lambd / (self.n_dim + self.lambd))

        for i in range(1, self.n_sig):
            self.covar_weights[i] = 1 / (2*(self.n_dim + self.lambd))
            self.mean_weights[i] = 1 / (2*(self.n_dim + self.lambd))

        self.sigmas = self.get_sigmas()

    def get_sigmas(self):
        """generates sigma points"""
        ret = np.zeros((self.n_sig, self.n_dim))

        tmp_mat = (self.n_dim + self.lambd)*self.p

        # print spr_mat
        spr_mat = scipy.linalg.sqrtm(tmp_mat)

        ret[0] = self.x
        for i in range(self.n_dim):
            ret[i+1] = self.x + spr_mat[i]
            ret[i+1+self.n_dim] = self.x - spr_mat[i]

        return ret.T

    def update(self, states, data, r_matrix):
        """
        this function does a measurement update
        :param states: list of indices of which states were measured, that is, which are being updated
        :param data: list of the data corresponding to the values in states
        :param r_matrix: error matrix for the data, again corresponding to the values in states
        :return:
        """

        num_states = len(states)

        # create y, sigmas of just the states that are being updated
        sigmas_split = np.split(self.sigmas, self.n_dim)
        y = np.concatenate([sigmas_split[i] for i in states])

        # create y_mean, the mean of just the states that are being updated
        x_split = np.split(self.x, self.n_dim)
        y_mean = np.concatenate([x_split[i] for i in states])

        # differences in y from y mean
        y_diff = deepcopy(y)
        x_diff = deepcopy(self.sigmas)
        for i in range(self.n_sig):
            for j in range(num_states):
                y_diff[j][i] -= y_mean[j]
            for j in range(self.n_dim):
                x_diff[j][i] -= self.x[j]

        # covariance of measurement
        P_yy = np.zeros((num_states,num_states))
        for i, val in enumerate(np.array_split(y_diff, self.n_sig, 1)):
            P_yy += self.covar_weights[i] * val.dot(val.T)

        # add measurement noise
        P_yy += r_matrix

        # covariance of measurement with states
        P_xy = np.zeros((self.n_dim, num_states))
        for i, val in enumerate(zip(np.array_split(y_diff, self.n_sig, 1), np.array_split(x_diff, self.n_sig, 1))):
            P_xy += self.covar_weights[i] * val[1].dot(val[0].T)

        k = np.dot(P_xy, np.linalg.inv(P_yy))

        y_actual = data

        self.x += np.dot(k, (y_actual - y_mean))
        self.p -= np.dot(k, np.dot(P_yy, k.T))
        # P = np.absolute(P)

        self.sigmas = self.get_sigmas()
        return

    def predict(self, timestep):
        """
        performs a prediction step
        :param timestep: float, amount of time since last update
        """
        sigmas_out = np.array([self.iterate(x, timestep) for x in self.sigmas.T]).T

        x_out = np.zeros(self.n_dim)

        # for each variable in X
        for i in range(self.n_dim):
            # the mean of that variable is the sum of
            # the weighted values of that variable for each iterated sigma point
            x_out[i] = sum((self.mean_weights[j] * sigmas_out[i][j] for j in range(self.n_sig)))

        p_out = np.zeros((self.n_dim, self.n_dim))
        # for each sigma point
        for i in range(self.n_sig):
            # take the distance from the mean
            # make it a covariance by multiplying by the transpose
            # weight it using the calculated weighting factor
            # and sum
            diff = sigmas_out.T[i] - x_out
            diff = np.atleast_2d(diff)
            p_out += self.covar_weights[i] * np.dot(diff.T, diff)

        # add process noise
        p_out += self.q

        self.sigmas = sigmas_out
        self.x = x_out
        self.p = p_out

    def get_weights(self):
        """
        :return: mean weights (n_sig x 1), covariance weights (n_sig x 1)
        """
        return self.mean_weights, self.covar_weights

    def get_state(self):
        """
        :return: current state (n_dim x 1)
        """
        return self.x

    def get_covar(self):
        """
        :return: current state covariance (n_dim x n_dim)
        """
        return self.p
