#!/usr/bin/env python

from ukf import UKF
import csv
import numpy as np
import matplotlib.pyplot as plt


def iterate_x(x_in, timestep):
    '''this function is based on the x_dot and can be nonlinear as needed'''
    ret = np.empty_like(x_in)
    ret[0:1,:] = x_in[0:1,:] + timestep * x_in[3:4,:] * np.cos(x_in[2:3,:])
    ret[1:2,:] = x_in[1:2,:] + timestep * x_in[3:4,:] * np.sin(x_in[2:3,:])
    ret[2:3,:] = x_in[2:3,:] + timestep * x_in[4:5,:]
    ret[3:4,:] = x_in[3:4,:] + timestep * x_in[5:6,:]
    ret[4:5,:] = x_in[4:5,:]
    ret[5:6,:] = x_in[5:6,:]
    return ret


def main():
    np.set_printoptions(precision=3)

    # Init Process Noise
    q = np.diag([0.0001,0.0001,0.0004,0.0025,0.0025,0.0025])

    # create measurement noise covariance matrices
    r_matrix = np.diag([0.01 , 0.03 , 0.02 , 0.001])
    # pass all the parameters into the UKF!
    # number of state variables, process noise, initial state, initial coariance, three tuning paramters, and the iterate function
    state_estimator = UKF(6, q, np.zeros((6,1)), 0.0001*np.eye(6), 0.04, 0.0, 2.0, iterate_x)

    # create three placehold for plotting
    real , estimate , time = [] , [] ,[]  
    with open('example.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        last_time = 0
        # read data
        for row in reader:
            row = [float(x) for x in row]

            cur_time = row[0]
            d_time = cur_time - last_time
            time.append(cur_time)
            
            #  x pos , y pos , heading , long velocity , yaw_rate , acc
            real_state = np.array([row[i] for i in [5, 6, 4, 3, 2, 1]]).reshape(-1,1)
            real.append(real_state)
            
            # create an array for the data from each sensor
            #             imu_yaw_rate , imu_accel , compass_hdg , encoder_vel
            measure = np.array([row[8] ,   row[7] ,     row[9] ,   row[10]]).reshape(-1,1)
            
            # update timestamp
            last_time = cur_time

            # prediction is pretty simple
            state_estimator.predict(d_time)

            # updating isn't bad either
            # remember that the updated states should be zero-indexed
            # the states should also be in the order of the noise and data matrices
            state_estimator.update(measure , r_matrix)

            # print ("--------------------------------------------------------")
            # print ("Real state: ", real_state.T)
            # print ("Estimated state: ", state_estimator.get_state().T)
            # print ("Difference: ", (real_state - state_estimator.get_state()).T)
            estimate.append(state_estimator.get_state())
    
    #  x pos , y pos , heading , long velocity , yaw_rate , acc        
    real = np.array(real)
    estimate = np.array(estimate)
    time = np.array(time)
    
    fig, axs = plt.subplots(5,1)
    axs[0].plot(real[:,0] , real[:,1] , "*-", alpha = 1, label='real' )
    axs[0].plot(estimate[:,0] , estimate[:,1] , "o-", alpha = 1, label='estimate')
    axs[0].set_title("Position estimation")
    axs[0].legend()
    
    axs[1].plot(time , real[:,2] , "*-", alpha = 1, label='real')
    axs[1].plot(time , estimate[:,2] , "-.", alpha = 1, label='estimate')
    axs[1].set_title("Heading estimation")
    axs[1].legend()
    
    axs[2].plot(time , real[:,3] , "*-", alpha = 1, label='real')
    axs[2].plot(time , estimate[:,3] , "-.", alpha = 1, label='estimate')
    axs[2].set_title("Long velocity estimation")
    axs[2].legend()
    
    axs[3].plot(time , real[:,4] , "*-", alpha = 1, label='real')
    axs[3].plot(time , estimate[:,4] , "-.", alpha = 1, label='estimate')
    axs[3].set_title("Heading rate estimation")
    axs[3].legend()
    
    axs[4].plot(time , real[:,5] , "*-", alpha = 1, label='real')
    axs[4].plot(time , estimate[:,5] , "-.", alpha = 1, label='estimate')
    axs[4].set_title("Acceleration estimation")
    axs[4].legend()
    
    plt.show()
if __name__ == "__main__":
    main()
