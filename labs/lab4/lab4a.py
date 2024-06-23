"""
Lab 4A - LIDAR Safety Stop
"""

import sys
import cv2 as cv
import numpy as np

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils


rc = racecar_core.create_racecar()

FRONT_WINDOW = (-10, 10)
REAR_WINDOW = (170, 190)

override = False
hax = 0.0



def start():
    rc.drive.stop()

def update():

    global override
    global hax

    hax += rc.get_delta_time()

    rt = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
    lt = rc.controller.get_trigger(rc.controller.Trigger.LEFT)
    speed = rt - lt

    # Calculate the distance in front of and behind the car
    scan = rc.lidar.get_samples()
    _, forward_dist = rc_utils.get_lidar_closest_point(scan, FRONT_WINDOW)
    _, back_dist = rc_utils.get_lidar_closest_point(scan, REAR_WINDOW)
    _, cone_dist = rc_utils.get_lidar_closest_point(scan, (170, 220))

    _, backL35_dist = rc_utils.get_lidar_closest_point(scan, (155, 175))
    _, backR35_dist = rc_utils.get_lidar_closest_point(scan, (185, 205))

    if rc.controller.is_down(rc.controller.Button.RB) or rc.controller.is_down(rc.controller.Button.LB):
        override = True
    if override and not rc.controller.is_down(rc.controller.Button.RB) or not rc.controller.is_down(rc.controller.Button.LB):
        override = False

    
    if backL35_dist > 350 or backR35_dist > 350:
        print("CONE")
        if cone_dist < 30:
            rc.drive.stop()
            speed = 0
        elif cone_dist < 50:
            speed = 1
        elif cone_dist < 80:
            speed += 0.75
        elif cone_dist < 120:
            speed += 0.5
    elif forward_dist < back_dist:
        if forward_dist < 30:
            rc.drive.stop()
            speed = 0
        elif forward_dist < 50:
            speed = -1
        elif forward_dist < 80:
            speed -= 0.75
        elif forward_dist < 120:
            speed -= 0.5
    else:
        if back_dist < 30:
            rc.drive.stop()
            speed = 0
        elif back_dist < 50:
            speed = 1
        elif back_dist < 80:
            speed += 0.75
        elif back_dist < 120:
            speed += 0.5

    #print(forward_dist)
    #print(back_dist)
    #print(backL35_dist)
    #print(backR35_dist)
    print(cone_dist)
    print(backR35_dist)
    print("\n")


    angle = rc.controller.get_joystick(rc.controller.Joystick.LEFT)[0]
    
    speed = rc_utils.clamp(speed, -1, 1)

    rc.drive.set_speed_angle(speed, angle)

    if rc.controller.is_down(rc.controller.Button.A):
        print("Speed:", speed, "Angle:", angle)

    if rc.controller.is_down(rc.controller.Button.B):
        print("Forward distance:", forward_dist, "Back distance:", back_dist)

    rc.display.show_lidar(scan)



if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
