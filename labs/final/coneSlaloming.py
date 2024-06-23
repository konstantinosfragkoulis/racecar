"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020
Phase 1 Challenge - Cone Slaloming
"""

########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv
import numpy as np
import math as mt
sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils
from enum import IntEnum

class State(IntEnum):
    search  = 0
    approach = 1
    LiDAR = 2
    turn = 3
    

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

curr_state: State = State.search

speed = 0
angle = 0

#HSV Values
BLUE = ((100, 175, 200), (130, 255, 255))  # The HSV range for the color blue
RED  = ((165, 180, 150),(179, 255, 255))

#Distance threshold before turning
DIST = 80

CAM_WIDTHd2 = rc.camera.get_width() >> 1
CAM_WIDTH = rc.camera.get_width()

OFFSET = 150

curState = State.search
curConeColor = None
coneDist = 9999
########################################################################################
# Functions
########################################################################################

# offset(x) ~= -(8/9)x + 335.56
# but this is faster to compute
def offset(x):
    return rc_utils.clamp(250-x, 0, CAM_WIDTH)

def findRedCone(image):
    if image is None:
        print("*** IMAGE IS NONE ***")
        return None
    else:
        return rc_utils.get_largest_contour(rc_utils.find_contours(image, RED[0], RED[1]), 800)

def findBlueCone(image):
    if image is None:
        print("*** IMAGE IS NONE ***")
        return None
    else:
        return rc_utils.get_largest_contour(rc_utils.find_contours(image, BLUE[0], BLUE[1]), 800)

def start():
    """
    This function is run once every time the start button is pressed
    """
    # Have the car begin at a stop
    rc.drive.stop()
    # Print start message
    print(">> Phase 1 Challenge: Cone Slaloming")

def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    global angle, speed
    global curState
    global curConeColor
    global coneDist

    rc.drive.set_max_speed(0.15)

    image = rc.camera.get_color_image()
    depthImage = rc.camera.get_depth_image()
    accel = rc.physics.get_linear_acceleration()

    redCone = findRedCone(image)
    blueCone = findBlueCone(image)

    # False -> red
    # True -> blue
    if curState == State.search or curState == State.turn:
        if redCone is not None and blueCone is not None:
            if rc_utils.get_contour_area(redCone) > rc_utils.get_contour_area(blueCone):
                curConeColor = False
                coneCenter = rc_utils.get_contour_center(redCone)
                if coneCenter is not None:
                    rc_utils.draw_circle(image, coneCenter)
            else:
                print("Blue cone is closer than red cone")
                curConeColor = True
                coneCenter = rc_utils.get_contour_center(blueCone)
                if coneCenter is not None:
                    rc_utils.draw_circle(image, coneCenter)
        elif redCone is None and blueCone is not None:
            print("The car can only see a blue cone")
            curConeColor = True
            coneCenter = rc_utils.get_contour_center(blueCone)
            if coneCenter is not None:
                rc_utils.draw_circle(image, coneCenter)
                print("Blue cone center Dist:", depthImage[coneCenter[0]][coneCenter[1]])
            else:
                print("Blue cone center is None")
        elif blueCone is None and redCone is not None:
            curConeColor = False
            coneCenter = rc_utils.get_contour_center(redCone)
            if coneCenter is not None:
                rc_utils.draw_circle(image, coneCenter)
        else:
            #curConeColor = None
            coneCenter = (None, None)
    else:
        if curConeColor == False: # red
            coneCenter = rc_utils.get_contour_center(redCone)
        elif curConeColor == True: # blue
            coneCenter = rc_utils.get_contour_center(blueCone)
    
    print(coneCenter, curConeColor)
    if redCone is not None:
        print("Red Cone Area: ", rc_utils.get_contour_area(redCone))
    if blueCone is not None:
        print("Blue Cone Area: ", rc_utils.get_contour_area(blueCone))

    if depthImage is not None and coneCenter is not None and not (None in coneCenter):
        coneDist = depthImage[coneCenter[0]][coneCenter[1]]
        print(coneDist)
        if coneDist > DIST:# and (curState == State.turn or curState == State.search):
            curState = State.approach
    
    if curState == State.approach:
        if coneCenter is not None:
            rc_utils.draw_circle(image, coneCenter)
            if curConeColor == False: # red
                if coneCenter[1] + OFFSET >= CAM_WIDTH:
                    rc_utils.draw_circle(image, (coneCenter[0], CAM_WIDTH-1))
                else:
                    rc_utils.draw_circle(image, (coneCenter[0], coneCenter[1] + OFFSET))
                angle = rc_utils.remap_range(coneCenter[1] + offset(depthImage[coneCenter[0]][coneCenter[1]]), 0, CAM_WIDTH, -1, 1, True)
                
                if angle < 0.9 and angle > -0.9:
                    # Its time to switch to the LiDAR to track the cone
                    # The car will move towards the cone and when the cone is at the correct position the car will turn
                    curState = State.LiDAR
            elif curConeColor == True: # blue
                if coneCenter[1] - OFFSET < 0:
                    rc_utils.draw_circle(image, (coneCenter[0], 0))
                else:
                    rc_utils.draw_circle(image, (coneCenter[0], coneCenter[1] - OFFSET))
                angle = rc_utils.remap_range(coneCenter[1] - offset(depthImage[coneCenter[0]][coneCenter[1]]), 0, CAM_WIDTH, -1, 1, True)
               
                if angle < 0.9 and angle > -0.9:
                    # Its time to switch to the LiDAR to track the cone
                    # The car will move towards the cone and when the cone is at the correct position the car will turn
                    curState = State.LiDAR

    elif curState == State.LiDAR:
        scan = rc.lidar.get_samples()

        coneAngle, coneDist = rc_utils.get_lidar_closest_point(scan)
        print("Cone Dist (LiDAR):", coneDist)
        print("Cone Angle (LiDAR):", coneAngle)
        if coneDist < 100: # The LiDAR can see the cone
            if coneAngle < 70 and coneAngle > 55: # The cone is at the correct position
                curState = State.turn
                # Turn correctly based on the color of the cone
            elif coneAngle < 310 and coneAngle > 285:
                curState = State.turn
                # Turn correctly based on the color of the cone
    elif curState == State.turn:
        # Look for cones
        # If there are none, keep turning left/right
        # The cone must be clone enough
        #angle = 0.75
        if coneCenter is not None and not (None in coneCenter) and depthImage[coneCenter[0]][coneCenter[1]] < 120:
            curState = State.approach
        else:
            if curConeColor == False: # red
                angle = -0.65
                if accel[1] > -9.75:
                    angle = -0.8
            else:
                angle = 0.65
                if accel[1] > -9.75:
                    angle = 0.8
    
    rc.display.show_color_image(image)

    speed = 1
    if accel[1] > -9.75:
        rc.drive.set_max_speed(0.1)
        speed = 0.15
    rc.drive.set_speed_angle(speed, angle)

    print("State:", curState)
    print("Cur Cone Color:", curConeColor)

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
