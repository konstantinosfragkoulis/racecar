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

# Add any global variables here
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
    ((130, 150, 150), (140, 255, 255), "purple"),
    ((10, 50, 50), (20, 255, 255), "orange"),
    ((170, 50, 50), (10, 255, 255), "red"),
    ((40, 50, 50), (80, 255, 255), "green"),
    ((100, 150, 50), (110, 255, 255), "blue")
]

CAMERA_HEIGHT = rc.camera.get_height()
CAMERA_WIDTH = rc.camera.get_width()
CAMERA_HEIGHTd2 = CAMERA_HEIGHT >> 1
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

# Necessary
speed = 0
angle = 0
curState: State = None
curConeSlalomingState: coneSlalomingStates = coneSlalomingStates.search

# Line Following
priorityColor = None
priorityColor2 = None

# Lane Following
laneColor = None
switchLane = None
contourSizeP = None
contourSizeO = None
### HAX - VERY IMPORTANT - START ###
hax = 0
### HAX - VERY IMPORTANT - END ###
curLaneColor = None

# Cone Slaloming
MIN_CONTOUR_AREA = 800
#Distance threshold before turning
DIST = 80
# Cone recovery
RECOVER_BLUE = False
RECOVER_RED = False
#Speed constants
APPROACH_SPEED = 1
TURN_SPEED = 0.65
#Angle constants
#RECOVER_ANGLE = 0.95
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

FRONT_WINDOW = (-5, 5)
LEFT_WINDOW = (265, 275) # Degrees not indices of the array scan
RIGHT_WINDOW = (85, 95) # Degrees not indices of the array scan
LEFT45_WINDOW = (310, 320)
RIGHT45_WINDOW = (40, 50)
LEFT45_WINDOW2 = (305, 325)
RIGHT45_WINDOW2 = (35, 55)
LEFT_WINDOW3 = (205, 25) # Degrees not indices of the array scan
RIGHT_WINDOW3 = (26, 204) # Degrees not indices of the array scan

#LEFT45_WINDOW = (260, 280) # Degrees not indices of the array scan
#RIGHT45_WINDOW = (80, 100) # Degrees not indices of the array scan
CENTER0_WINDOW = (350, 10) # Degrees not indices of the array scan

### HAX4 - VERY IMPORTANT - START ###
hax4 = 0
hax5 = 0 # hax for 2nd turn
hax6 = 0 # hax for 2nd turn
haxFlag1 = False
haxFlag2 = False
### HAX4 - VERY IMPORTANT - END ###

hasTurnedRight = False

firstTurn = False

timer = -2
counter = 0

hasChangedStateLiDAR = False

trueSpeed = 0

TEN_PERCENT = 1/10
RAD_TO_DEG = 180 / math.pi
DEG_TO_RAD = math.pi / 180

########################################################################################
# Functions
########################################################################################

def followLine():
    global speed
    global angle
    global priorityColor
    global image

    rc.drive.set_max_speed(0.3)

    if image is None:
        contourCenter = None
        print("**COLOR IMAGE IS NONE**")
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
            print("**CONTOUR IS NONE**")

        if contour is not None:
            contourCenter = rc_utils.get_contour_center(contour)

            rc_utils.draw_contour(colorImg, contour)
            rc_utils.draw_circle(colorImg, contourCenter)
            rc.display.show_color_image(colorImg)
        else:
            contourCenter = None
        
        if contourCenter is not None:
            angle = rc_utils.clamp(rc_utils.remap_range(contourCenter[1], 0, 640, -1, 1) * 1.6, -1, 1)

            linearAccel = rc.physics.get_linear_acceleration()

            speed = rc_utils.clamp(1 - 0.5*abs(angle), -1, 1)
            if linearAccel[1] > -9.72:
                speed = 0.2

def detectContours(colorImg):
    global laneColor

    if colorImg is None:
        print("**COLOR IMAGE IS NONE**")
    else:
        contourP = rc_utils.get_largest_contour(rc_utils.find_contours(colorImg, PURPLE[0], PURPLE[1]))
        contourO = rc_utils.get_largest_contour(rc_utils.find_contours(colorImg, ORANGE[0], ORANGE[1]))

        if contourP is not None:
            contourSizeP = rc_utils.get_contour_area(contourP)
        else:
            contourSizeP = None
        if contourO is not None:
            contourSizeO = rc_utils.get_contour_area(contourO)
        else:
            contourSizeO = None

        if contourSizeP is not None and contourSizeO is not None:
            if laneColor == "purple":
                return (rc_utils.get_contour_center(contourP), rc_utils.get_contour_center(contourO), contourSizeP > contourSizeO+200)
            else:
                return (rc_utils.get_contour_center(contourP), rc_utils.get_contour_center(contourO), contourSizeP+200 > contourSizeO)
        elif contourSizeP is None:
            return (None, rc_utils.get_contour_center(contourO), False)
        elif contourSizeO is None:
            return (rc_utils.get_contour_center(contourP), None, True)
        else:
            print("**CONTOUR IS NONE**")
            return (None, None, None)

def findCone(isRed: bool):
    global image

    if image is None:
        return None
    # Returns the center of the closest red and blue cone
    return (rc_utils.get_contour_center(rc_utils.get_largest_contour(rc_utils.find_contours(image, PURPLE[0], PURPLE[1]))), rc_utils.get_contour_center(rc_utils.get_largest_contour(rc_utils.find_contours(image, ORANGE[0], ORANGE[1]))))

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

    rc.drive.set_max_speed(0.275)

    speed = 1

    colorImg1 = rc_utils.crop(image, CROP_RIGHT[0], CROP_RIGHT[1])
    colorImg2 = rc_utils.crop(image, CROP_LEFT[0], CROP_LEFT[1])
    colorImg3 = rc_utils.crop(image, CROP_FLOOR_LINE[0], CROP_FLOOR_LINE[1])
    (contourCenterP1, contourCenterO1, contourPBiggerThanContourO1) = detectContours(colorImg1) #right
    (contourCenterP2, contourCenterO2, contourPBiggerThanContourO2) = detectContours(colorImg2) #left

    # If the lane that the car is following (or should follow) is purple
    if curLaneColor == "purple":
        # If there are 2 purple lines detected
        if contourCenterP1 is not None and contourCenterP2 is not None and not (None in contourCenterP1) and not (None in contourCenterP2):
            # The angles for the 2 images (The color image is cut in half)
            angle1 = rc_utils.remap_range(contourCenterP1[1], 0, CAMERA_WIDTHd2, -1, 1)
            angle2 = rc_utils.remap_range(contourCenterP2[1], 0, CAMERA_WIDTHd2, -1, 1)

            rc_utils.draw_circle(colorImg3, (contourCenterP1[0], contourCenterP1[1]+(CAMERA_WIDTH//2)))
            rc_utils.draw_circle(colorImg3, contourCenterP2)

            rc.display.show_color_image(colorImg3)

            if switchLane == None:
                angle = rc_utils.clamp(angle1 + angle2, -1, 1)
                #speed = rc_utils.clamp(1 - 0.5*abs(angle), -1, 1)
            elif switchLane == 'left':
                angle = -0.2
            else:
                angle = 0.2

        # If 2 purple lines are not detected
        elif (contourCenterP1 is None or (None in contourCenterP1)) or (contourCenterP2 is None or (None in contourCenterP2)):
            #print("Less than 2 purple lines detected")
            # If atleast 1 orange line is detected, make the car follow the oragne lane
            if (contourCenterO1 is not None and not (None in contourCenterO1)) or (contourCenterO2 is not None and not (None in contourCenterO2)):
                curLaneColor = "orange"
            
    # If the lane that the car is following (or should follow) is orange
    elif curLaneColor == "orange":
        # If there are 2 orange lines detected
        if contourCenterO1 is not None and contourCenterO2 is not None and not (None in contourCenterO1) and not (None in contourCenterO2):
            angle1 = rc_utils.remap_range(contourCenterO1[1], 0, CAMERA_WIDTHd2, -1, 1)
            angle2 = rc_utils.remap_range(contourCenterO2[1], 0, CAMERA_WIDTHd2, -1, 1)

            rc_utils.draw_circle(colorImg3, (contourCenterO1[0], contourCenterO1[1]+(CAMERA_WIDTH//2)))
            rc_utils.draw_circle(colorImg3, contourCenterO2)

            rc.display.show_color_image(colorImg3)
            
            if switchLane == None:
                angle = rc_utils.clamp(angle1 + angle2, -1, 1)
                #speed = rc_utils.clamp(1 - 0.5*abs(angle), -1, 1)
            elif switchLane == 'left':
                angle = -0.2
            else:
                angle = 0.2
    
        # If 2 orange lines are not detected
        elif (contourCenterO1 is None or (None in contourCenterO1)) or (contourCenterO2 is None or (None in contourCenterO2)):
            #print("Less than 2 orange lines detected")
            # If atleast 1 purple line is detected, make the car follow the purple lane
            if (contourCenterP1 is not None and not (None in contourCenterP1)) or (contourCenterP2 is not None and not (None in contourCenterP2)):
                curLaneColor = "purple"

    if not firstTurn and not haxFlag2 and curLaneColor != laneColor:
        angle = -1
        print("first turn")
    elif firstTurn and haxFlag2 and curLaneColor != laneColor:
        angle = 1
        print("second turn")

def findGate():
    global image

    if image is None:
        print("**COLOR IMAGE IS NONE**")
    else:
        contourR = rc_utils.get_largest_contour(rc_utils.find_contours(image, CONE_RED[0], CONE_RED[1]), MIN_CONTOUR_AREA)
        contourB = rc_utils.get_largest_contour(rc_utils.find_contours(image, CONE_BLUE[0], CONE_BLUE[1]), MIN_CONTOUR_AREA)

        if contourR is not None:
            contourCenterR = rc_utils.get_contour_center(contourR)
            rc_utils.draw_circle(image, contourCenterR)
        else:
            contourCenterR = None
        if contourB is not None:
            contourCenterB = rc_utils.get_contour_center(contourB)
            rc_utils.draw_circle(image, contourCenterB)
        else:
            contourCenterB = None
        rc.display.show_color_image(image)

        # Return the center of the clostest red and blue cone
        return (contourCenterR, contourCenterB)

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

    # If the car sees the white ramp, switch state to findGate
    if hax3 < 10:
        hax3 += rc.get_delta_time()
    else:
        centerRow = image[CAMERA_WIDTHd2]
        for i in range(0, CAMERA_WIDTH):
            if centerRow[i][0] > 214 and centerRow[i][1] > 195 and centerRow[i][2] > 175 and depthImg[CAMERA_WIDTHd2][i] < 100:
                curConeSlalomingState = coneSlalomingStates.gate
                print("GATES")

    # Slalom cones

    if curConeSlalomingState == coneSlalomingStates.search:
        angle = -0.5
        if RECOVER_RED: 
            angle = -RECOVER_ANGLE
        elif RECOVER_BLUE:
            angle = RECOVER_ANGLE
        if redDepth < blueDepth and redDepth!=0:
            curConeSlalomingState = coneSlalomingStates.approachRed
        elif blueDepth < redDepth and blueDepth!=0:
            curConeSlalomingState  = coneSlalomingStates.approachBlue
        elif redDepth != 0:
            curConeSlalomingState = coneSlalomingStates.approachRed
        elif blueDepth !=0:
            curConeSlalomingState = coneSlalomingStates.approachBlue  
    elif curConeSlalomingState == coneSlalomingStates.approachRed:
        if RECOVER_BLUE: RECOVER_BLUE = False
        if redDepth == 0.0: 
            curConeSlalomingState = coneSlalomingStates.search
        elif redDepth < DIST: 
            curConeSlalomingState = coneSlalomingStates.turnRed  
        else:
            rc_utils.draw_circle(image,redCenter)
            angle = rc_utils.remap_range(redCenter[1], 0, CAMERA_WIDTH, -1,1, True) 
    elif curConeSlalomingState == coneSlalomingStates.approachBlue:
        if RECOVER_RED: RECOVER_RED = False
        if blueDepth == 0.0: 
            curConeSlalomingState = coneSlalomingStates.search
        elif blueDepth < DIST: 
            curConeSlalomingState = coneSlalomingStates.turnBlue
        else:
            rc_utils.draw_circle(image, blueCenter)
            angle = rc_utils.remap_range(blueCenter[1], 0, CAMERA_WIDTH, -1,1,True) 
    elif curConeSlalomingState == coneSlalomingStates.turnRed:
        hax2 += rc.get_delta_time()
        if hax2 < 0.85:
            # angle = rc_utils.remap_range(redDepth,150,0,0,1, True)
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
            # angle = rc_utils.remap_range(blueDepth,150,0,0,-1, True)
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

    rc.display.show_color_image(image)

def trackObstacle():
    global oldObstacleAngle
    global newObstacleAngle
    global oldObstacleDist
    global newObstacleDist

    oldObstacleDist = newObstacleDist
    oldObstacleAngle = newObstacleAngle

    extendedScan = np.concatenate((scan, scan))

    for i in range(700, 740):
        if abs(extendedScan[i+oldObstacleAngle] - oldObstacleDist) < oldObstacleDist*0.2:
            newObstacleAngle = i+oldObstacleAngle
            if newObstacleAngle > 719:
                newObstacleAngle -= 719
            elif newObstacleAngle < 0:
                newObstacleAngle += 719
            newObstacleDist = extendedScan[i+oldObstacleAngle]

    """flag = None

    for i in range(20, -20):
        if i+oldObstacleAngle > 0:
            if abs(scan[i+oldObstacleAngle] - oldObstacleDist) < oldObstacleDist*0.2:
                newObstacleAngle = i+oldObstacleAngle
                newObstacleDist = scan[i+oldObstacleAngle]
        else:
            flag = i
            break
    
    if flag is not None:
        for i in range(flag, -20):
            if abs(scan[i+oldObstacleAngle] - oldObstacleDist) < oldObstacleDist*0.2:
                newObstacleAngle = i+oldObstacleAngle
                newObstacleDist = scan[i+oldObstacleAngle]
    """
    print("New Angle:", newObstacleAngle)
    print("New Dist:", newObstacleDist)

def followWalls():
    print("FOLLOW WALLS")

    global speed
    global angle
    global scan
    global hasChangedStateLiDAR

    rc.drive.set_max_speed(0.25)

    if not hasChangedStateLiDAR:
        largestDist = 0
        for i in range(-15, 16):
            if largestDist < scan[1]:
                largestDist = scan[i]

        _start = None
        _end = None
        for i in range(-10, 11):
            if _start == None:
                if largestDist > scan[i] - scan[i] * TEN_PERCENT and largestDist < scan[i] + scan[i] * TEN_PERCENT:
                    _start = i
            if _end == None:
                if largestDist > scan[-i] - scan[-i] * TEN_PERCENT and largestDist < scan[-i] + scan[-i] * TEN_PERCENT:
                    _end = -i

        _center = (_start+_end) / 2

    speed = 0.5
    
    if not hasChangedStateLiDAR and scan[0] > 220:
        angle = rc_utils.remap_range(_center, -50, 50, -1, 1, True) + 0.05
    else:
        hasChangedStateLiDAR = True
        
        _, left45Dist = rc_utils.get_lidar_closest_point(scan, LEFT45_WINDOW2)
        _, right45Dist = rc_utils.get_lidar_closest_point(scan, RIGHT45_WINDOW2)

        if left45Dist > right45Dist:
            angle = rc_utils.clamp(right45Dist - left45Dist + 5, -1, 0)
        else:
            angle = rc_utils.clamp(right45Dist - left45Dist - 5, 0, 1)

        speed = rc_utils.clamp(1.25 - angle, -1, 1)
    
def start():
    """
    This function is run once every time the start button is pressed
    """
    global speed
    global angle

    speed = 0
    angle = 0

    # Have the car begin at a stop
    rc.drive.stop()

    # Print start message
    print(">> Final Challenge - Time Trial")

def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
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
    global trueSpeed

    image = rc.camera.get_color_image()
    depthImg = rc.camera.get_depth_image()
    scan = rc.lidar.get_samples()
    markers = rc_utils.get_ar_markers(image)
    linearAccel = rc.physics.get_linear_acceleration() # Not global
    trueSpeed += linearAccel[2] * rc.get_delta_time()

    print("True Speed:", trueSpeed)

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

        # Marker indicating the start of Line Following
        if mkId == 0 and mkOrientation == rc_utils.Orientation.LEFT:
            if priorityColor is None:
                priorityColor = mkColor
            curState = State.lineFollowing

        # Marker indicating which color is NOT the second priorityColor
        if mkId == 0 and mkOrientation == rc_utils.Orientation.RIGHT:
            if {mkColor, priorityColor} == {"blue", "red"}:
                priorityColor2 = "green"
            if {mkColor, priorityColor} == {"green", "red"}:
                priorityColor2 = "blue"
            if {mkColor, priorityColor} == {"blue", "green"}:
                priorityColor2 = "red"

        # Marker indicating the start of Lane Following
        elif mkId == 1 and mkOrientation == rc_utils.Orientation.UP:
            # Change priority color
            # The line changes color when the car can see this AR Marker
            if mkDist < 500:
                priorityColor = priorityColor2

            # When the car is close enough to the marker it starts Lane Following
            if mkDist < 200:
                laneColor = mkColor
                curLaneColor = laneColor
                curState = State.laneFollowing 
        
        # Marker indicating to switch lane
        elif mkId == 199 and mkDist < 180:
            # Count how many seconds have passed after starting to switch lane
            hax = 0
            if mkOrientation == rc_utils.Orientation.LEFT:
                switchLane = 'left'
            else:
                switchLane = 'right'
        
        # Marker indicating to start Cone Slaloming
        elif mkId == 2:
            if mkDist < 80:
                curState = State.coneSlaloming
            #curState = State.coneSlaloming

        # Marker indicating to start Wall Following
        elif mkId == 3 and mkOrientation == rc_utils.Orientation.UP:
            if mkDist < 100:
                curState = State.wallFollowing
            elif mkDist < 200:
                angle = rc_utils.remap_range(mkCenter[1] - 50, 0, CAMERA_WIDTH, -1, 1)
            

    # If the car is switching lane, start counting
    if switchLane is not None:
        hax += rc.get_delta_time()

        if hax > 1:
            switchLane = None
    
    # Make the car move
    rc.drive.set_speed_angle(speed, angle)

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()