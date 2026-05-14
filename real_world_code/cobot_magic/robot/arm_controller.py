import math
import threading
import numpy as np
import rospy

from sensor_msgs.msg import JointState
from std_msgs.msg import Header

import config


class AutomatedController:
    """Dual-arm joint controller. Publishes target joint states over ROS topics."""

    def __init__(self):
        # --- Publishers ---
        self.left_arm_pub = rospy.Publisher('/master/joint_left', JointState, queue_size=10)
        self.right_arm_pub = rospy.Publisher('/master/joint_right', JointState, queue_size=10)

        # --- Subscribers and state ---
        self.left_arm_current_pos = []
        self.right_arm_current_pos = []
        self.left_arm_state_lock = threading.Lock()
        self.right_arm_state_lock = threading.Lock()
        rospy.Subscriber('/puppet/joint_left', JointState, self.left_arm_callback)
        rospy.Subscriber('/puppet/joint_right', JointState, self.right_arm_callback)

        # --- Control parameters ---
        self.publish_rate = 40
        self.arm_steps_length = (
            np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2]) * config.ARM_SPEED
        ).tolist()
        self.JOINT_NAMES = [f'joint{i}' for i in range(7)]

        # --- Target state and thread control ---
        self.target_left_joints = []
        self.target_right_joints = []
        self.arm_control_thread = None
        self.run_thread = threading.Event()

        self.wait_for_initial_state()
        self.start_control_thread()

    def left_arm_callback(self, msg):
        with self.left_arm_state_lock:
            self.left_arm_current_pos = list(msg.position)

    def right_arm_callback(self, msg):
        with self.right_arm_state_lock:
            self.right_arm_current_pos = list(msg.position)

    def wait_for_initial_state(self):
        rospy.loginfo("Waiting for initial arm states...")
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            with self.left_arm_state_lock, self.right_arm_state_lock:
                if self.left_arm_current_pos and self.right_arm_current_pos:
                    self.target_left_joints = self.left_arm_current_pos[:]
                    self.target_right_joints = self.right_arm_current_pos[:]
                    rospy.loginfo("Successfully received initial arm states.")
                    return
            rospy.loginfo("Waiting for /puppet/joint_left and /puppet/joint_right messages...")
            rate.sleep()

    def _arm_control_loop(self):
        """Background loop that continuously publishes joint commands."""
        rate = rospy.Rate(self.publish_rate)
        while self.run_thread.is_set() and not rospy.is_shutdown():
            with self.left_arm_state_lock, self.right_arm_state_lock:
                local_target_left = self.target_left_joints[:]
                local_target_right = self.target_right_joints[:]
                current_left = self.left_arm_current_pos[:]
                current_right = self.right_arm_current_pos[:]

            if not all([local_target_left, local_target_right, current_left, current_right]):
                rate.sleep()
                continue

            next_left_pos = current_left[:]
            next_right_pos = current_right[:]

            for i in range(7):
                diff_l = local_target_left[i] - current_left[i]
                if abs(diff_l) > self.arm_steps_length[i]:
                    next_left_pos[i] += self.arm_steps_length[i] * math.copysign(1, diff_l)
                else:
                    next_left_pos[i] = local_target_left[i]

                diff_r = local_target_right[i] - current_right[i]
                if abs(diff_r) > self.arm_steps_length[i]:
                    next_right_pos[i] += self.arm_steps_length[i] * math.copysign(1, diff_r)
                else:
                    next_right_pos[i] = local_target_right[i]

            left_msg = JointState(name=self.JOINT_NAMES, position=next_left_pos,
                                  header=Header(stamp=rospy.Time.now()))
            self.left_arm_pub.publish(left_msg)

            right_msg = JointState(name=self.JOINT_NAMES, position=next_right_pos,
                                   header=Header(stamp=rospy.Time.now()))
            self.right_arm_pub.publish(right_msg)
            rate.sleep()

    def start_control_thread(self):
        if not self.arm_control_thread or not self.arm_control_thread.is_alive():
            self.run_thread.set()
            self.arm_control_thread = threading.Thread(target=self._arm_control_loop, daemon=True)
            self.arm_control_thread.start()
            rospy.loginfo("Arm control thread started.")

    def stop_control_thread(self):
        self.run_thread.clear()
        if self.arm_control_thread:
            self.arm_control_thread.join(timeout=1)
        rospy.loginfo("Arm control thread stopped.")

    def move_to_goal(self, left_target, right_target, tolerance=0.05, timeout=30):
        """Set new target joints and block until reached or until timeout."""
        rospy.loginfo("Setting new arm goal.")
        with self.left_arm_state_lock, self.right_arm_state_lock:
            self.target_left_joints = left_target
            self.target_right_joints = right_target

        start_time = rospy.Time.now()
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and (rospy.Time.now() - start_time).to_sec() < timeout:
            with self.left_arm_state_lock, self.right_arm_state_lock:
                current_l = self.left_arm_current_pos[:]
                current_r = self.right_arm_current_pos[:]

            if not all([current_l, current_r]):
                rate.sleep()
                continue

            left_reached = all(abs(left_target[i] - current_l[i]) < tolerance for i in range(7))
            right_reached = all(abs(right_target[i] - current_r[i]) < tolerance for i in range(7))

            if left_reached and right_reached:
                rospy.loginfo("Arms have reached the goal position.")
                return True
            rate.sleep()

        rospy.logwarn("Timeout reached while moving arms to goal.")
        return False
