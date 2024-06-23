"""
Lab 4B - LIDAR Wall Following
"""

import sys
import cv2 as cv
import numpy as np

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils


rc = racecar_core.create_racecar()

LEFT_WINDOW = (260, 280)
RIGHT_WINDOW = (80, 100)
LEFT45_WINDOW = (305, 325)
RIGHT45_WINDOW = (35, 55)

speed = 0.0
angle = 0.0


def start():
    rc.drive.stop()

def update():
    
    global speed
    global angle

    # TODO: Follow the wall to the right of the car without hitting anything.

    #rt = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
    #lt = rc.controller.get_trigger(rc.controller.Trigger.LEFT)
    #speed = rt - lt

    scan = rc.lidar.get_samples()

    _, leftDist = rc_utils.get_lidar_closest_point(scan, LEFT_WINDOW)
    _, rightDist = rc_utils.get_lidar_closest_point(scan, RIGHT_WINDOW)
    _, left45Dist = rc_utils.get_lidar_closest_point(scan, LEFT45_WINDOW)
    _, right45Dist = rc_utils.get_lidar_closest_point(scan, RIGHT45_WINDOW)

    print("LEFT:")
    print(leftDist)
    print("RIGHT:")
    print(rightDist)
    print("LEFT45:")
    print(left45Dist)
    print("RIGHT45:")
    print(right45Dist)
    print("\n")


    if left45Dist > right45Dist:
        angle = rc_utils.clamp(right45Dist - left45Dist + 5, -1, 0)
    elif right45Dist > left45Dist:
        angle = rc_utils.clamp(right45Dist - left45Dist - 5, 0, 1)
    elif leftDist > rightDist:
        angle = rc_utils.clamp(rightDist - leftDist + 1, -1, 0)
    elif rightDist > leftDist:
        angle = rc_utils.clamp(rightDist - leftDist - 1, 0, 1)

    #angle = rc.controller.get_joystick(rc.controller.Joystick.LEFT)[0]
    speed = rc_utils.clamp(1.25 - angle, -1, 1)

    rc.drive.set_speed_angle(speed, angle)



if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
