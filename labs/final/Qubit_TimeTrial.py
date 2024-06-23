"""
Copyright MIT and Harvey Mudd College
MIT License
Fall 2020

Final Challenge - Time Trial
"""

########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv
import numpy as np
from enum import IntEnum
import math

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

class State(IntEnum):
    lineFollowing = 0
    laneFollowing = 1
    coneSlaloming = 2
    wallFollowing = 3

class coneSlalomingStates(IntEnum):
    search  = 0
    approachRed = 1
    approachBlue = 2
    turnRed = 3
    turnBlue = 4
    gate = 5

potentialColors = [
    ((130, 150, 150), (140, 255, 255), False),
    ((10, 50, 50), (20, 255, 255), True),
    ((170, 50, 50), (10, 255, 255), "red"),
    ((40, 50, 50), (80, 255, 255), "green"),
    ((100, 150, 50), (110, 255, 255), "blue")
]

CAMERA_HEIGHT = rc.camera.get_height()
CAMERA_WIDTH = rc.camera.get_width()
CAMERA_WIDTHd2 = CAMERA_WIDTH >> 1

CROP_FLOOR_LINE = ((300, 0), (CAMERA_HEIGHT, CAMERA_WIDTH))
CROP_RIGHT = ((300, CAMERA_WIDTHd2), (CAMERA_HEIGHT, CAMERA_WIDTH))
CROP_LEFT = ((300, 0), (CAMERA_HEIGHT, CAMERA_WIDTHd2))

PURPLE = ((130, 150, 150), (140, 255, 255))
ORANGE = ((10, 150, 150), (20, 255, 255))
RED = ((170, 150, 150), (10, 255, 255))
GREEN = ((55, 150, 150), (65, 255, 255))
BLUE = ((100, 100, 100), (110, 255, 255))

CONE_RED = ((165, 0, 0),(179, 255, 255))
CONE_BLUE = ((100, 175, 200), (130, 255, 255))

speed = 0
angle = 0
curState: State = None
curConeSlalomingState: coneSlalomingStates = coneSlalomingStates.search

priorityColor = None
priorityColor2 = None

laneColor: bool = None
switchLane: bool = None
contourSizeP = None
contourSizeO = None
### HAX - VERY IMPORTANT - START ###
hax = 0
### HAX - VERY IMPORTANT - END ###
curLaneColor: bool = None

MIN_CONTOUR_AREA = 800
DIST = 80
RECOVER_BLUE = False
RECOVER_RED = False
APPROACH_SPEED = 1
TURN_SPEED = 0.65
RECOVER_ANGLE = 1
TURN_ANGLE =  1
### HAX2,3 - VERY IMPORTANT - START ###
hax2 = 0
hax3 = 0
### HAX2,3 - VERY IMPORTANT - END ###

image = None
depthImg = None
scan = None

extremeCounter = 0
isInExtreme = False

LEFT45_WINDOW = (310, 320)
RIGHT45_WINDOW = (40, 50)
LEFT45_WINDOW2 = (305, 325)
RIGHT45_WINDOW2 = (35, 55)

### HAX4 - VERY IMPORTANT - START ###
hax4 = 0
hax5 = 0 # hax for 2nd turn
hax6 = 0 # hax for 2nd turn
haxFlag1 = False
haxFlag2 = False
### HAX4 - VERY IMPORTANT - END ###

firstTurn = False

counter = 0

hasChangedStateLiDAR = False

TEN_PERCENT = 1/10

linearAccel = None

hasStartedLiDAR = False
turnDirLiDAR = False
hax7 = 0
hasSeenARMarkerLiDAR = False
hax8 = 0
haxFlag3 = False

collisionDetected = False

dt = 0
########################################################################################
# Functions
########################################################################################

def followLine():
    global speed
    global angle
    global priorityColor
    global image
    global linearAccel
    global collisionDetected
    global dt

    if dt < 0.041:
        rc.drive.set_max_speed(0.45)
    elif dt < 0.055:
        rc.drive.set_max_speed(0.4)
    else:
        rc.drive.set_max_speed(0.3)

    if image is None:
        contourCenter = None
    else:
        colorImg = rc_utils.crop(image, CROP_FLOOR_LINE[0], CROP_FLOOR_LINE[1])

        contourR = rc_utils.get_largest_contour(rc_utils.find_contours(colorImg, RED[0], RED[1]))
        contourG = rc_utils.get_largest_contour(rc_utils.find_contours(colorImg, GREEN[0], GREEN[1]))
        contourB = rc_utils.get_largest_contour(rc_utils.find_contours(colorImg, BLUE[0], BLUE[1]))

        if priorityColor == 'red' and contourR is not None:
            contour = contourR
        elif priorityColor == 'green' and contourG is not None:
            contour = contourG
        elif priorityColor == 'blue' and contourB is not None:
            contour = contourB
        else:
            contour = None

        if contour is not None:
            contourCenter = rc_utils.get_contour_center(contour)
        else:
            contourCenter = None
        
        if contourCenter is not None:
            angle = rc_utils.clamp(rc_utils.remap_range(contourCenter[1], 0, 640, -1, 1) * 1.6, -1, 1)

            speed = rc_utils.clamp(1 - 0.5*abs(angle), -1, 1)
            if linearAccel[1] > -9.72:
                speed = 0.2

        closestPointAngle, closestPoint = rc_utils.get_lidar_closest_point(scan, (295, 65))
        if closestPoint < 20:
            collisionDetected = True
        
        if collisionDetected:
            if closestPoint < 20:
                speed = -1
                if closestPointAngle > 270:
                    angle = -1
                else:
                    angle = 1
            else:
                collisionDetected = False

def detectContours(_colorImg):
    if _colorImg is None:
        return (None, None)
    return (rc_utils.get_contour_center(rc_utils.get_largest_contour(rc_utils.find_contours(_colorImg, PURPLE[0], PURPLE[1]))), rc_utils.get_contour_center(rc_utils.get_largest_contour(rc_utils.find_contours(_colorImg, ORANGE[0], ORANGE[1]))))

def followLane():
    global speed
    global angle
    global laneColor
    global curLaneColor
    global switchLane
    global image
    global firstTurn
    global hax6
    global haxFlag2
    global dt

    if dt < 0.041:
        rc.drive.set_max_speed(0.3)
    elif dt < 0.065:
        rc.drive.set_max_speed(0.295)
    else:
        rc.drive.set_max_speed(0.275)

    speed = 1

    colorImg1 = rc_utils.crop(image, CROP_RIGHT[0], CROP_RIGHT[1])
    colorImg2 = rc_utils.crop(image, CROP_LEFT[0], CROP_LEFT[1])
    (contourCenterP1, contourCenterO1) = detectContours(colorImg1)
    (contourCenterP2, contourCenterO2) = detectContours(colorImg2)

    if curLaneColor == False:
        if contourCenterP1 is not None and contourCenterP2 is not None and not (None in contourCenterP1) and not (None in contourCenterP2):
            angle1 = rc_utils.remap_range(contourCenterP1[1], 0, CAMERA_WIDTHd2, -1, 1)
            angle2 = rc_utils.remap_range(contourCenterP2[1], 0, CAMERA_WIDTHd2, -1, 1)

            if switchLane == None:
                angle = rc_utils.clamp(angle1 + angle2, -1, 1)
            elif switchLane == False:
                angle = -0.2
            else:
                angle = 0.2

        elif (contourCenterP1 is None or (None in contourCenterP1)) or (contourCenterP2 is None or (None in contourCenterP2)):
            if (contourCenterO1 is not None and not (None in contourCenterO1)) or (contourCenterO2 is not None and not (None in contourCenterO2)):
                curLaneColor = True

    elif curLaneColor == True:
        if contourCenterO1 is not None and contourCenterO2 is not None and not (None in contourCenterO1) and not (None in contourCenterO2):
            angle1 = rc_utils.remap_range(contourCenterO1[1], 0, CAMERA_WIDTHd2, -1, 1)
            angle2 = rc_utils.remap_range(contourCenterO2[1], 0, CAMERA_WIDTHd2, -1, 1)
            
            if switchLane == None:
                angle = rc_utils.clamp(angle1 + angle2, -1, 1)
            elif switchLane == False:
                angle = -0.2
            else:
                angle = 0.2

        elif (contourCenterO1 is None or (None in contourCenterO1)) or (contourCenterO2 is None or (None in contourCenterO2)):
            if (contourCenterP1 is not None and not (None in contourCenterP1)) or (contourCenterP2 is not None and not (None in contourCenterP2)):
                curLaneColor = False

    if not firstTurn and not haxFlag2 and curLaneColor != laneColor:
        angle = -1
    elif firstTurn and haxFlag2 and curLaneColor != laneColor:
        angle = 1

def findGate():
    global image

    if image is None:
        return (None, None)
    return (rc_utils.get_contour_center(rc_utils.get_largest_contour(rc_utils.find_contours(image, CONE_RED[0], CONE_RED[1]), MIN_CONTOUR_AREA)), rc_utils.get_contour_center(rc_utils.get_largest_contour(rc_utils.find_contours(image, CONE_BLUE[0], CONE_BLUE[1]), MIN_CONTOUR_AREA)))

def slalomCones():
    global angle, speed, RECOVER_BLUE, RECOVER_RED
    global curConeSlalomingState, hax2, hax3
    global DIST, RED, BLUE, MIN_CONTOUR_AREA
    global image
    global depthImg
    global extremeCounter, isInExtreme

    rc.drive.set_max_speed(0.25)

    redContour = rc_utils.get_largest_contour(rc_utils.find_contours(image, CONE_RED[0], CONE_RED[1]), MIN_CONTOUR_AREA)
    blueContour = rc_utils.get_largest_contour(rc_utils.find_contours(image, CONE_BLUE[0], CONE_BLUE[1]), MIN_CONTOUR_AREA)

    redCenter = rc_utils.get_contour_center(redContour) if redContour is not None else None
    blueCenter = rc_utils.get_contour_center(blueContour) if blueContour is not None else None
    
    redDepth = depthImg[redCenter[0]][redCenter[1]] if redContour is not None else 0.0
    blueDepth = depthImg[blueCenter[0]][blueCenter[1]] if blueContour is not None else 0.0

    if hax3 < 10:
        hax3 += rc.get_delta_time()
    else:
        centerRow = image[CAMERA_WIDTHd2]
        for i in range(0, CAMERA_WIDTH):
            if centerRow[i][0] > 214 and centerRow[i][1] > 195 and centerRow[i][2] > 175 and depthImg[CAMERA_WIDTHd2][i] < 100:
                curConeSlalomingState = coneSlalomingStates.gate

    if curConeSlalomingState == coneSlalomingStates.search:
        angle = -0.5
        if RECOVER_RED: 
            angle = -RECOVER_ANGLE
        elif RECOVER_BLUE:
            angle = RECOVER_ANGLE
        if redDepth < blueDepth and redDepth != 0:
            curConeSlalomingState = coneSlalomingStates.approachRed
        elif blueDepth < redDepth and blueDepth != 0:
            curConeSlalomingState  = coneSlalomingStates.approachBlue
        elif redDepth != 0:
            curConeSlalomingState = coneSlalomingStates.approachRed
        elif blueDepth != 0:
            curConeSlalomingState = coneSlalomingStates.approachBlue  
    elif curConeSlalomingState == coneSlalomingStates.approachRed:
        if RECOVER_BLUE: RECOVER_BLUE = False
        if redDepth == 0.0: 
            curConeSlalomingState = coneSlalomingStates.search
        elif redDepth < DIST: 
            curConeSlalomingState = coneSlalomingStates.turnRed  
        else:
            angle = rc_utils.remap_range(redCenter[1], 0, CAMERA_WIDTH, -1, 1, True) 
    elif curConeSlalomingState == coneSlalomingStates.approachBlue:
        if RECOVER_RED: RECOVER_RED = False
        if blueDepth == 0.0: 
            curConeSlalomingState = coneSlalomingStates.search
        elif blueDepth < DIST: 
            curConeSlalomingState = coneSlalomingStates.turnBlue
        else:
            angle = rc_utils.remap_range(blueCenter[1], 0, CAMERA_WIDTH, -1, 1, True) 
    elif curConeSlalomingState == coneSlalomingStates.turnRed:
        hax2 += rc.get_delta_time()
        if hax2 < 0.85:
            angle = TURN_ANGLE
        elif hax2 < 1: 
            angle = 0
        else:
            hax2 = 0
            RECOVER_RED = True
            curConeSlalomingState = coneSlalomingStates.search
    elif curConeSlalomingState == coneSlalomingStates.turnBlue:
        hax2 += rc.get_delta_time()
        if hax2 < 0.85:
            angle = -TURN_ANGLE
        elif hax2 < 1: 
            angle = 0
        else:
            hax2 = 0
            RECOVER_BLUE = True
            curConeSlalomingState = coneSlalomingStates.search
    elif curConeSlalomingState == coneSlalomingStates.gate:
        (contourCenterR, contourCenterB) = findGate()

        if contourCenterR is not None and contourCenterB is not None and not (None in contourCenterR) and not (None in contourCenterB):
            if isInExtreme:
                isInExtreme = False
                extremeCounter += 1
            CENTER = ((contourCenterR[0] + contourCenterB[0]) / 2, (contourCenterR[1] + contourCenterB[1]) / 2)
        elif contourCenterR is not None and not (None in contourCenterR):
            if isInExtreme:
                isInExtreme = False
                extremeCounter += 1
            CENTER = (contourCenterR[0], contourCenterR[1] + 200)
        elif contourCenterB is not None and not (None in contourCenterB):
            if isInExtreme:
                isInExtreme = False
                extremeCounter += 1
            CENTER = (contourCenterB[0], contourCenterB[1] - 200)
        else:
            isInExtreme = True
            if extremeCounter != 2:
                CENTER = (0, 0)
            else:
                CENTER = (0, CAMERA_WIDTH)
        angle = rc_utils.remap_range(CENTER[1], 0, CAMERA_WIDTH, -1, 1, True)

    if curConeSlalomingState == (coneSlalomingStates.approachBlue or coneSlalomingStates.approachRed or coneSlalomingStates.search):
        speed = APPROACH_SPEED
    else:
        speed = TURN_SPEED

def followWalls():
    global speed
    global angle
    global scan
    global hasChangedStateLiDAR
    global turnDirLiDAR
    global hasSeenARMarkerLiDAR
    global hax8
    global haxFlag3
    global collisionDetected
    global dt

    if dt < 0.035:
        rc.drive.set_max_speed(0.325)
    elif dt < 0.055:
        rc.drive.set_max_speed(0.3)
    elif dt < 0.075:
        rc.drive.set_max_speed(0.275)
    else:
        rc.drive.set_max_speed(0.25)

    if not hasChangedStateLiDAR:
        largestDist = 0
        for i in range(-15, 16):
            largestDist = max(largestDist, scan[i])

        _center = 0
        _centerd = 0
        for i in range(-15, 16):
            if largestDist > scan[i] - scan[i] * TEN_PERCENT and largestDist < scan[i] + scan[i] * TEN_PERCENT:
                _center += i
                _centerd += 1

        _center //= _centerd

    speed = 0.5
    
    if not hasChangedStateLiDAR and scan[0] > 220:
        angle = rc_utils.remap_range(_center, -50, 50, -1, 1, True) + 0.05
    else:
        hasChangedStateLiDAR = True

        if hasSeenARMarkerLiDAR:
            hax8 += rc.get_delta_time()

        if hasSeenARMarkerLiDAR and hax8 > 5:
            left45Dist = 0
            for i in range(LEFT45_WINDOW2[0], LEFT45_WINDOW2[1]):
                left45Dist = max(left45Dist, scan[i<<1])

            right45Dist = 0
            for i in range(RIGHT45_WINDOW2[0], RIGHT45_WINDOW2[1]):
                right45Dist = max(right45Dist, scan[i<<1])
        else:
            (_, left45Dist) = rc_utils.get_lidar_closest_point(scan, LEFT45_WINDOW2)
            (_, right45Dist) = rc_utils.get_lidar_closest_point(scan, RIGHT45_WINDOW2)

        if left45Dist > right45Dist:
            angle = rc_utils.clamp(right45Dist - left45Dist + 5, -1, 0)
        else:
            angle = rc_utils.clamp(right45Dist - left45Dist - 5, 0, 1)

        speed = rc_utils.clamp(1.5 - angle, -1, 1)

        if turnDirLiDAR == True:
            angle = 1
        elif turnDirLiDAR == False:
            angle = -1

        closestPointAngle, closestPoint = rc_utils.get_lidar_closest_point(scan, (295, 65))
        if closestPoint < 20:
            collisionDetected = True
        
        if collisionDetected:
            if closestPoint < 20:
                speed = -1
                if closestPointAngle > 270:
                    angle = -1
                else:
                    angle = 1
            else:
                collisionDetected = False

def start():
    global speed
    global angle

    speed = 0
    angle = 0

    rc.drive.stop()

def update():
    global speed
    global angle
    global curState
    global priorityColor
    global priorityColor2
    global laneColor
    global curLaneColor
    global switchLane
    global hax
    global image
    global depthImg
    global scan
    global hax5
    global firstTurn
    global hax6
    global haxFlag1
    global haxFlag2
    global linearAccel
    global hasStartedLiDAR
    global turnDirLiDAR
    global hax7
    global hasSeenARMarkerLiDAR
    global dt

    image = rc.camera.get_color_image()
    depthImg = rc.camera.get_depth_image()
    scan = rc.lidar.get_samples()
    markers = rc_utils.get_ar_markers(image)
    linearAccel = rc.physics.get_linear_acceleration()
    dt = rc.get_delta_time()

    if curState == State.lineFollowing:
        followLine()
    elif curState == State.laneFollowing:
        followLane()
    elif curState == State.coneSlaloming:
        slalomCones()
    elif curState == State.wallFollowing:
        followWalls()

    if angle < -0.5:
        hax5 += rc.get_delta_time()
    if hax5 > 2.5:
        firstTurn = True
        haxFlag1 = True
    if haxFlag1:
        hax6 += rc.get_delta_time()
        if hax6 > 2:
            haxFlag2 = True

    switchLane = None

    for marker in markers:
        marker.detect_colors(image, potentialColors)

        mkId = marker.get_id()
        mkColor = marker.get_color()
        mkCorners = marker.get_corners()
        mkOrientation = marker.get_orientation()
        mkCenter = ((mkCorners[0][0] + mkCorners[2][0]) >> 1, (mkCorners[0][1] + mkCorners[2][1]) >> 1)
        mkDist = depthImg[mkCenter[0]][mkCenter[1]]

        if mkId == 0 and mkOrientation == rc_utils.Orientation.LEFT:
            if priorityColor is None:
                priorityColor = mkColor
            curState = State.lineFollowing
        if mkId == 0 and mkOrientation == rc_utils.Orientation.RIGHT:
            if {mkColor, priorityColor} == {"blue", "red"}:
                priorityColor2 = "green"
            if {mkColor, priorityColor} == {"green", "red"}:
                priorityColor2 = "blue"
            if {mkColor, priorityColor} == {"blue", "green"}:
                priorityColor2 = "red"
        elif mkId == 1 and mkOrientation == rc_utils.Orientation.UP:
            if mkDist < 500:
                priorityColor = priorityColor2
            if mkDist < 200:
                laneColor = mkColor
                curLaneColor = laneColor
                curState = State.laneFollowing
        elif mkId == 199:
            if hasStartedLiDAR and mkDist < 80 and mkOrientation == rc_utils.Orientation.LEFT:
                hax7 = 0
                turnDirLiDAR = False
                hasSeenARMarkerLiDAR = True
            elif hasStartedLiDAR and mkDist < 93 and mkOrientation == rc_utils.Orientation.RIGHT:
                hax7 = 0
                turnDirLiDAR = True
                hasSeenARMarkerLiDAR = True
            elif mkDist < 180:
                hax = 0
                if mkOrientation == rc_utils.Orientation.LEFT:
                    switchLane = False
                else:
                    switchLane = True
        elif mkId == 2:
            if mkDist < 80:
                curState = State.coneSlaloming
        elif mkId == 3 and mkOrientation == rc_utils.Orientation.UP:
            if mkDist < 100:
                curState = State.wallFollowing
                hasStartedLiDAR = True
            elif mkDist < 200:
                angle = rc_utils.remap_range(mkCenter[1] - 50, 0, CAMERA_WIDTH, -1, 1)

    if switchLane != None:
        hax += rc.get_delta_time()

        if hax > 1:
            switchLane = None
    elif turnDirLiDAR != None:
        hax7 += rc.get_delta_time()

        if hax7 > 0.9:
            turnDirLiDAR = None
    
    rc.drive.set_speed_angle(speed, angle)

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()