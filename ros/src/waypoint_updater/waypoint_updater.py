#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

from scipy.spatial import KDTree
import numpy as np
import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 100 # Number of waypoints we will publish. You can change this number
PUBLISHING_RATE = 50
MAX_DECEL = 0.5

def get_waypoint_velocity(waypoint):
    return waypoint.twist.twist.linear.x

def set_waypoint_velocity(waypoint, velocity):
    waypoint.twist.twist.linear.x = velocity


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.base_waypoints = None
        self.waypoints_2d = None
        self.kdtree = None
        self.pose = None
        self.stop_line_wp = -1

        self.loop()
        rospy.spin()


    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


    def loop(self):
        rate = rospy.Rate(PUBLISHING_RATE)
        while not rospy.is_shutdown():
            if self.pose and self.kdtree:
                closest_point_index = self.get_closest_waypoint_idx()
                self.publish_waypoints(closest_point_index)
            rate.sleep()


    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx =  self.kdtree.query([x,y],1)[1]

        closest_coord = self.waypoints_2d[closest_idx]
        pre_coord = self.waypoints_2d[closest_idx-1]

        cl_vect = np.array(closest_coord)
        prev_vect = np.array(pre_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)

        return closest_idx


    def publish_waypoints(self,closest_idx):
        lane = Lane()

        end_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_waypoints[closest_idx:end_idx]

        if (self.stop_line_wp == -1 or self.stop_line_wp > end_idx):
            lane.waypoints = base_waypoints
        else:
            dist_to_light = self.stop_line_wp - closest_idx
            lane.waypoints = self.decelerate(base_waypoints, dist_to_light)
#             rospy.loginfo("dist to light {} and speed before {} after {}".format(dist_to_light, \
#                                                                                  get_waypoint_velocity(base_waypoints[0]), \
#                                                                                  get_waypoint_velocity(lane.waypoints[0])))
        self.final_waypoints_pub.publish(lane)


    def decelerate(self, waypoints, dist_to_light):
        lane_wps = []
        dist_to_stop = max(0, dist_to_light - 2)
        for i in range(len(waypoints)):
            wp = Waypoint()
            wp.pose = waypoints[i].pose
            dist = self.distance(waypoints, i, dist_to_stop)
            dec_vel = math.sqrt(2 * MAX_DECEL * dist)
            if dec_vel < 1:
                dec_vel = 0.0
            set_waypoint_velocity(wp, min(dec_vel, get_waypoint_velocity(wp)))
            lane_wps.append(wp)
        return lane_wps


    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        self.base_waypoints = waypoints.waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[pt.pose.pose.position.x, pt.pose.pose.position.y] for pt in waypoints.waypoints]
            self.kdtree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        if (msg.data != self.stop_line_wp):
            self.stop_line_wp = msg.data
            rospy.loginfo("received stop line wp idx {}".format(msg.data))

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
