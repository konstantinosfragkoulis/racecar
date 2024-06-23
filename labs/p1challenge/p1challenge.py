"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020
Phase 1 Challenge - Cone Slaloming
"""

#######################################################################################
#`Imports
#######################################################################################

import sys
import cv2 as cv
import numpy as np
from enum import IntEnum
import math

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils

#######################################################################################
#Global variables
#######################################################################################

rc = racecar_core.create_racecar()

#Add any global variables here

class State(IntEnum):
    findCone = 0
    coneSlaloming = 1
    recover = 2
    avoidCollision = 3


RED = ((170, 150, 150), (10, 255, 255))
BLUE = ((105, 150, 150), (120, 255, 255))

CAMERA_WIDTH = rc.camera.get_width()
CAMERA_HEIGHT = rc.camera.get_height()

RAD_2_DEG = 180/math.pi

speed = 0
angle = 0

curState = State.findCone

d = 10  #Some number
#######################################################################################
#Functions
#######################################################################################

def findConeLidar():
    scan = rc.lidar.get_samples()
    for i in range(-160, 160):
        if scan[i] != 0:
            phi_idx = i
            break
    for i in range(-160, 160):
        if scan[i] < scan[phi_idx] and scan[i] != 0:
            phi_idx = i
    print("Cone Dist:", scan[phi_idx])
    print("Phi_idx:", phi_idx)
    return phi_idx

def trackCone(oldPhi_idx, oldConeDist):
    scan = rc.lidar.get_samples()
    for i in range(-5, 5):
        if scan[oldPhi_idx+i] < oldConeDist and oldConeDist-oldConeDist/10 < scan[oldPhi_idx+i]:
            return oldPhi_idx+i

def lostCone(oldConeDist):
    scan = rc.lidar.get_samples()
    for i in range(-200, -161):
        if scan[i] < oldConeDist and oldConeDist-oldConeDist/10 < scan[i]:
            return True
    for i in range(161, 200):
        if scan[i] < oldConeDist and oldConeDist-oldConeDist/10 < scan[i]:
            return True
    return False

def findAngleLidar(phi_idx, coneColor):
    global d

    scan = rc.lidar.get_samples()
    phi = phi_idx/2
    c = 1000
    for i in range(-2, 2):
        if scan[phi_idx+i] < c and scan[phi_idx+i] != 0:
            c = scan[phi_idx+i]
    c = (scan[phi_idx-1] + scan[phi_idx] + scan[phi_idx+1]) / 3
    c = scan[phi_idx]
    C = (c * math.sin(phi), c * math.cos(phi))
    #d = 10 # Some number

    if coneColor == "blue":  #If cone is blue, the car has to go left
        W = (C[0]-d, C[1])
    elif coneColor == "red":  #If cone is red, the car has to go right
        W = (C[0]+d, C[1])

    w = math.sqrt(W[0]**2 + W[1]**2)
    print("Phi =", phi)
    print("Phi_idx =", phi_idx)
    for i in range(-15, 15):
        print("Scan", phi_idx+i, ":", scan[phi_idx+i])
    print("C = (", C[0], ",", C[1], ")")
    print("W = (", W[0], ",", W[1], ")")
    print("w =", w)
    print("c =", c)
    theta = math.acos((d**2 - w**2 - c**2) / (2*c*w)) * RAD_2_DEG
    print("Theta:", theta)

    if coneColor == "blue":
        print("Cone color: Blue")
        print("Angle sum:", phi-theta)
        angle = rc_utils.remap_range(phi - theta, -90, 90, -1, 1, 1)
    else:
        print("Cone color: Red")
        print("Angle sum:", phi+theta)
        angle = rc_utils.remap_range(phi + theta, -90, 90, -1, 1, 1)

    print("ANGLE:", angle)
    return angle, phi_idx, c

def findConeColor():

    colorImg = rc.camera.get_color_image()

    if colorImg is None:
        print("**COLOR IMAGE IS NONE**")
    else:
        contourR = rc_utils.get_largest_contour(rc_utils.find_contours(colorImg, RED[0], RED[1]))
        contourB = rc_utils.get_largest_contour(rc_utils.find_contours(colorImg, BLUE[0], BLUE[1]))

        if contourR is not None and contourB is not None:
            if rc_utils.get_contour_area(contourR) > rc_utils.get_contour_area(contourB):
                coneColor = "red"
            else:
                coneColor = "blue"
        elif contourR is not None and contourB is None:
            coneColor = "red"
        elif contourB is not None and contourR is None:
            coneColor = "blue"
        else:
            print("**CONTOUR IS NONE**")

    return coneColor

def findCones():

    colorImg = rc.camera.get_color_image()
    depthImg = rc.camera.get_depth_image()

    if colorImg is None:
        contourCenter = None
        print("**COLOR IMAGE IS NONE**")
    else:
        contourR = rc_utils.get_largest_contour(rc_utils.find_contours(colorImg, RED[0], RED[1]))
        contourB = rc_utils.get_largest_contour(rc_utils.find_contours(colorImg, BLUE[0], BLUE[1]))

        if contourR is not None and contourB is not None:
            if rc_utils.get_contour_area(contourR) > rc_utils.get_contour_area(contourB):
                contour = contourR
                coneColor = "red"
            else:
                contour = contourB
                coneColor = "blue"
        elif contourR is not None and contourB is None:
            contour = contourR
            coneColor = "red"
        elif contourB is not None and contourR is None:
            contour = contourB
            coneColor = "blue"
        else:
            contour = None
            print("**CONTOUR IS NONE**")

        if contour is not None:
            contourCenter = rc_utils.get_contour_center(contour)

            rc_utils.draw_contour(colorImg, contour)
            rc_utils.draw_circle(colorImg, contourCenter)
            rc.display.show_color_image(colorImg)
        else:
            contourCenter = None

    return (contourCenter, coneColor)

def findAngle(contourCenter, coneColor):
    global d

    phi = rc_utils.remap_range(contourCenter[1], 0, CAMERA_WIDTH, -21, 21)
    phi_idx = int(2*phi)
    scan = rc.lidar.get_samples()
    c = (scan[phi_idx-1] + scan[phi_idx] + scan[phi_idx+1]) / 3
    
    #c = scan[phi_idx]
    C = (c * math.sin(phi), c * math.cos(phi))
    # d = 100 # Some number

    if coneColor == "blue": # If cone is blue, the car has to go left
        W = (C[0]-d, C[1])
    elif coneColor == "red": # If cone is red, the car has to go right
        W = (C[0]+d, C[1])

    w = math.sqrt(W[0]**2 + W[1]**2)
    print("Cone x:", contourCenter[1])
    print("Phi =", phi)
    print("Phi_idx =", phi_idx)
    for i in range(-15, 15):
        print("Scan", phi_idx+i, ":", scan[phi_idx+i])
    print("C = (", C[0], ",", C[1], ")")
    print("W = (", W[0], ",", W[1], ")")
    print("w =", w)
    print("c =", c)
    theta = math.acos((d**2 - w**2 - c**2) / (2*c*w)) * RAD_2_DEG
    print("Theta:", theta)

    if coneColor == "blue":
        print("Cone color: Blue")
        print("Angle sum:", phi-theta)
        angle = rc_utils.remap_range(phi - theta, -90, 90, -1, 1, 1)
    else:
        print("Cone color: Red")
        print("Angle sum:", phi+theta)
        angle = rc_utils.remap_range(phi + theta, -90, 90, -1, 1, 1)

    print("ANGLE:", angle)
    return angle

def start():
    """
    This function is run once every time the start button is pressed
    """
    #Have the car begin at a stop
    rc.drive.stop()

    #Print start message
    print(">> Phase 1 Challenge: Cone Slaloming")


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    global angle
    global curState
    #TODO: Slalom between red and blue cones.  The car should pass to the right of
    #each red cone and the left of each blue cone.
    
    if curState == State.findCone:
        (angle, phi_idx, coneDist) = findAngleLidar(findConeLidar(), findConeColor())
        curState = State.coneSlaloming
    elif curState == State.coneSlaloming:
        if lostCone(coneDist):
            curState = State.findCone
            return
        (contourCenter, coneColor) = findCones()
        (angle, phi_idx, coneDist) = findAngleLidar(trackCone(phi_idx, coneDist))
    if curState == State.findCones:
        (contourCenter, coneColor) = findCones()
        angle = findAngle(contourCenter, coneColor)
        curState = State.coneSlaloming
    elif curState == State.coneSlaloming:
        (contourCenter, coneColor) = findCones()
        angle = findAngle(contourCenter, coneColor)
    elif curState == State.revocer:
        pass
    elif curState == State.avoidCollision:
        pass
        
    rc.drive.set_speed_angle(0.75, angle)

#######################################################################################
#DO NOT MODIFY: Register start and update and begin execution
#######################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
