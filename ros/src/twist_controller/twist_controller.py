#!/usr/bin/env python

from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704
KP = 0.2
KI = 0.04
KD = 0.02
MIN_NUM = 0.0
MAX_NUM = 0.5
MIN_SPEED = 0.1
TAU = 0.5
TS = 0.02


class Controller(object):
    def __init__(self, vehicle_mass, decel_limit, accel_limit, wheel_radius,
                 wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        # TODO: Implement
        self.yaw_controller = YawController(wheel_base=wheel_base, 
                                            steer_ratio=steer_ratio, 
                                            min_speed=MIN_SPEED, 
                                            max_lat_accel=max_lat_accel, 
                                            max_steer_angle=max_steer_angle)
        self.throttle_controller = PID(kp=KP, ki=KI, kd=KD, mn=MIN_NUM, mx=MAX_NUM)
        self.lowpass_filter = LowPassFilter(tau=TAU, ts=TS)

        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.vehicle_mass = vehicle_mass
        self.wheel_radius = wheel_radius
        
        self.last_time = rospy.get_time()


    def control(self, dbw_enabled, current_vel, linear_vel, angular_vel):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0., 0., 0

        current_vel = self.lowpass_filter.filt(current_vel)

        cur_time = rospy.get_time()
        time_diff = cur_time - self.last_time
        self.last_time = cur_time

        vel_diff = linear_vel - current_vel

        throttle = self.throttle_controller.step(vel_diff, time_diff)
        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)
        brake = 0.0

        if linear_vel == 0.0 and current_vel < 0.1:
            brake = 600
            throttle = 0.0
        elif vel_diff < 0 and throttle < 0.1:
            throttle = 0.0
            decel = max(self.decel_limit, vel_diff)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius

        return throttle, brake, steering
