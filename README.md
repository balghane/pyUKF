# A general unscented kalman filter
Written by Basel Alghanem at the University of Michigan ROAHM Lab and based on "The Unscented Kalman Filter for Nonlinear Estimation" by Wan, E. A., & Van Der Merwe, R. (2000). This python unscented kalman filter (UKF) implementation supports multiple measurement updates (even simultaneously) and allows you to easily plug in your model and measurements!

# Examples
Trying out the first example (example.py) should be really easy. It reads data from a provided csv and demonstrates the core functionality in a simple case. The system being modeled could be some kind of driving robot that has three sensors: an IMU measuring linear accelerations and angular velocities, a compass, and encoders measuring longitudinal velocity. The provided example trajectory runs at a constant longitudinal acceleration and yaw rate. The challenge of this example is tracking x and y position when they're not being directly measured. The dynamic model is described by the following:  
![](Examples%20Files/latex_x.png)  
![](Examples%20Files/latex_x_dot.png)

We introduced gaussian noise with the following covariance:  
![](Examples%20Files/q.png)

From the compass and encoders, we're using the heading and longitudinal velocity. From the IMU, we're using yaw rate and longitudinal acceleration. We introduced gaussian noise for each sensor with these covariances: 
![](Examples%20Files/r_compass.png)  
![](Examples%20Files/r_encoder.png)  
![](Examples%20Files/r_imu.png)

These exact covariances were input to the UKF, so the filter tracks quite well despite the process and measurement noise. Much of the challenge of using a UKF or any kalman filter is determining the covariances for real systems. Hopefully this library eases the programming burden!
