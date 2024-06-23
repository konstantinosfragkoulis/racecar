"""
Final Challenge - Time Trial
"""

import sys
import cv2 as cv
import numpy as np
from enum import IntEnum

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils

rc = racecar_core.create_racecar()

class Part(IntEnum):
    lineFollowing = 0
    laneFollowing = 1
    laneSwitching = 2
    coneSlaloming = 3
    wallFollowing = 4
    searching = 5

class State(IntEnum):
    search = 0
    approach_red = 1
    approach_blue = 2
    turn_red = 3
    turn_blue = 4
    stop = 5

### Cone Slaloming Vars
coneState: State = State.search
MIN_CONTOUR_AREA = 500
DIST = 80
RECOVER_BLUE = False
RECOVER_RED = False
APPROACH_SPEED = 0.65
TURN_SPEED = 0.50
RECOVER_ANGLE = 0.65
TURN_ANGLE =  0.75
counter = 0.0

BLUE2 = ((100, 150, 150), (130, 255, 255))
RED2  = ((165, 0, 0),(179, 255, 255))

def get_contour(HSV, MIN_CONTOUR_AREA = 30):
    image = rc.camera.get_color_image()
    if image is None:
        return None
    else:
        contours = rc_utils.find_contours(image, HSV[0], HSV[1])
        contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)
        return contour
### Cone Slaloming Vars







potentialColors = [
    ((110, 59, 50), (165, 255, 255), "purple"),
    ((10, 50, 50), (20, 255, 255), "orange"),
    ((170, 50, 50), (10, 255, 255), "red"),
    ((40, 50, 50), (80, 255, 255), "green"),
    ((100, 150, 50), (110, 255, 255), "blue")
]

PURPLE = ((110, 59, 50), (165, 255, 255))
ORANGE = ((10, 50, 50), (20, 255, 255))
RED = ((170, 50, 50), (10, 255, 255))
GREEN = ((35, 50, 50), (80, 255, 255))
BLUE = ((100, 150, 50), (110, 255, 255))

LEFT_WINDOW = (260, 280)
RIGHT_WINDOW = (80, 100)
LEFT45_WINDOW = (305, 325)
RIGHT45_WINDOW = (35, 55)

CROP_FLOOR = ((300, 0), (rc.camera.get_height(), rc.camera.get_width()))

speed = 0.0
angle = 0.0

priorityColor = None
laneSpeedOverride = False
laneColor = ""

switchTimer = 0
switchSide = ""

contourP = None
contourO = None
hax2 = 0

hax = 0
leftDist = 0
rightDist = 0
left45Dist = 0
right45Dist = 0
wasFollowingWalls = False

curPart = Part.searching

def followLine():

    rc.drive.set_max_speed(0.5)

    global speed
    global angle
    global priorityColor

    followLineColorImg = rc.camera.get_color_image()

    if followLineColorImg is None:
        contourCenter = None
    else:
        followLineColorImg = rc_utils.crop(followLineColorImg, CROP_FLOOR[0], CROP_FLOOR[1])

        contoursR = rc_utils.find_contours(followLineColorImg, RED[0], RED[1])
        contoursG = rc_utils.find_contours(followLineColorImg, GREEN[0], GREEN[1])
        contoursB = rc_utils.find_contours(followLineColorImg, BLUE[0], BLUE[1])

        contourR = rc_utils.get_largest_contour(contoursR)
        contourG = rc_utils.get_largest_contour(contoursG)
        contourB = rc_utils.get_largest_contour(contoursB)

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

            rc_utils.draw_contour(followLineColorImg, contour)
            rc_utils.draw_circle(followLineColorImg, contourCenter)
        else:
            contourCenter = None

        rc.display.show_color_image(followLineColorImg)

    if contourCenter is not None:
        angle = rc_utils.remap_range(contourCenter[1], 0, 640, -1, 1, True)

    speed = rc_utils.clamp(1 - angle, -1, 1)

def followLane():

    rc.drive.set_max_speed(0.3)

    global speed
    global angle
    global laneColor
    global laneSpeedOverride
    global contourP
    global contourO
    global hax2
    global switchSide
    global curPart

    hax2 += rc.get_delta_time()

    followLaneColorImg = rc.camera.get_color_image()

    if followLaneColorImg is None:
        contourCenter = None
    else:
        followLaneColorImg = rc_utils.crop(followLaneColorImg, CROP_FLOOR[0], CROP_FLOOR[1])

        contoursP = rc_utils.find_contours(followLaneColorImg, PURPLE[0], PURPLE[1])
        contoursO = rc_utils.find_contours(followLaneColorImg, ORANGE[0], ORANGE[1])

        if laneColor == "orange":
            contourO = rc_utils.get_largest_contour(contoursO)
            contourP = rc_utils.get_largest_contour(contoursP)
            if contourO is not None:
                contour = contourO
            elif contourP is not None:
                contour = contourP
            else:
                contour = None
        elif laneColor == "purple":
            contourP = rc_utils.get_largest_contour(contoursP)
            contourO = rc_utils.get_largest_contour(contoursO)
            if contourP is not None:
                contour = contourP
            elif contourO is not None:
                contour = contourO
            else:
                contour = None

        if len(contoursP) > 0 and laneColor == "orange":
            laneSpeedOverride = True
            rc.drive.set_max_speed(0.25)
            speed = 0.25
            if angle > 0.01:
                angle = rc_utils.clamp(angle + 1, -1, 1)
        if len(contoursO) > 0 and laneColor == "purple":
            laneSpeedOverride = True
            rc.drive.set_max_speed(0.25)
            speed = 0.25
            if angle > 0.01:
                angle = rc_utils.clamp(angle + 1, -1, 1)

        if contour is not None:
            contourCenter = rc_utils.get_contour_center(contour)

            rc_utils.draw_contour(followLaneColorImg, contour)
            rc_utils.draw_circle(followLaneColorImg, contourCenter)
        else:
            contourCenter = None

        rc.display.show_color_image(followLaneColorImg)

    if switchSide == rc_utils.Orientation.LEFT: #42
        if hax2 >= 25.5:
            curPart = Part.coneSlaloming
    else: # 38
        if hax2 >= 22.75:
            curPart = Part.coneSlaloming

    if contourCenter is not None:
        angle = rc_utils.remap_range(contourCenter[1], 0, 640, -1, 1, True)

    if not laneSpeedOverride:
        speed = rc_utils.clamp(1 - angle, -1, 1)

    laneSpeedOverride = False

def switchLane():

    rc.drive.set_max_speed(0.5)

    global speed
    global angle
    global switchTimer
    global switchSide
    global wasFollowingWalls
    global hax2
    global curPart

    switchTimer += rc.get_delta_time()

    if not wasFollowingWalls:
        if switchTimer < 0.75:
            speed = 1
            if switchSide == rc_utils.Orientation.RIGHT:
                angle = 0.15
            else:
                angle = -0.15
        else:
            curPart = Part.laneFollowing
            switchTimer = 0
            hax2 = 0
    else:
        rc.drive.set_max_speed(0.25)
        if switchSide == rc_utils.Orientation.LEFT:
            if switchTimer < 1.45:
                speed = 0.35
                angle = -1
            else:
                curPart = Part.wallFollowing
        elif switchSide == rc_utils.Orientation.RIGHT:
            if switchTimer < 1.45:
                speed = 0.35
                angle = 1
            else:
                curPart = Part.wallFollowing

def slalomCones():
    global angle, speed, RECOVER_BLUE, RECOVER_RED
    global coneState, counter
    global DIST, RED2, BLUE2, MIN_CONTOUR_AREA
    global curPart

    print(curPart)

    image = rc.camera.get_color_image()
    depth_image_original = (rc.camera.get_depth_image() - 0.01) % 10000
    depth_image_original= cv.GaussianBlur(depth_image_original,(3,3),0)

    red_contour = get_contour(RED2, MIN_CONTOUR_AREA)
    blue_contour = get_contour(BLUE2, MIN_CONTOUR_AREA)

    red_center = rc_utils.get_contour_center(red_contour) if red_contour is not None else None
    blue_center = rc_utils.get_contour_center(blue_contour) if blue_contour is not None else None

    red_depth = depth_image_original[red_center[0]][red_center[1]] if red_contour is not None else 0.0
    blue_depth = depth_image_original[blue_center[0]][blue_center[1]] if blue_contour is not None else 0.0

    if coneState == State.search:
        if RECOVER_RED and (blue_depth > 20 or blue_depth == 0):
            angle = -RECOVER_ANGLE
        if RECOVER_BLUE and (red_depth > 20 or red_depth == 0):
            angle = RECOVER_ANGLE
        if red_depth < blue_depth and red_depth!=0:
            coneState = State.approach_red
        if blue_depth < red_depth and blue_depth!=0:
            coneState = State.approach_blue
        elif red_depth != 0:
            coneState = State.approach_red
        elif blue_depth !=0:
            coneState = State.approach_blue

        if coneState == State.approach_red:
            if RECOVER_BLUE: RECOVER_BLUE = False
            if red_depth == 0.0:
                coneState = State.search
            elif red_depth < DIST:
                coneState = State.search
            else:
                angle = rc_utils.remap_range(red_center[1], 0, rc.camera.get_width(), -1,1, True)

        if coneState == State.approach_blue:
            if RECOVER_RED: RECOVER_RED = False
            if blue_depth == 0.0:
                coneState = State.search
            elif blue_depth < DIST:
                coneState = State.turn_blue
            else:
                angle = rc_utils.remap_range(blue_center[1], 0, rc.camera.get_width(), -1,1,True)

        if coneState == State.turn_red:
            counter += rc.get_delta_time()
            if counter < 0.85:
                angle = TURN_ANGLE
            elif counter < 1:
                angle = 0
            else:
                counter = 0
                RECOVER_RED = True
                coneState = State.search

        if coneState == State.turn_blue:
            counter += rc.get_delta_time()
            if counter < 0.85:
                angle = -TURN_ANGLE
            elif counter < 1:
                angle = 0
            else:
                counter = 0
                RECOVER_BLUE = True
                coneState = State.search

            if coneState == (State.approach_blue or State.approach_red or State.search):
                speed = APPROACH_SPEED
            else:
                speed = TURN_SPEED

    print(f"S:{coneState} V{speed:.2f} Angle: {angle:.2f} Rec R:{RECOVER_RED} Rec B:{RECOVER_BLUE}")
    print(f"Red depth:{red_depth:.2F} Blue depth:{blue_depth:.2F}")

def followWalls():
    global speed
    global angle
    global hax
    global leftDist
    global rightDist
    global left45Dist
    global right45Dist
    global wasFollowingWalls

    wasFollowingWalls = True

    rc.drive.set_max_speed(0.25)

    hax += rc.get_delta_time()

    if hax < 1.5:
        speed = 0.75
        angle = 0.75
    elif hax < 2.45:
        speed = 0.75
        angle = -0.25
    elif hax < 4.75:
        speed = 1
        angle = 0
    elif hax < 4.9:
        speed = 1
        angle = -0.3
    elif hax < 7:
        speed = 1
        angle = 0

    scan = rc.lidar.get_samples()

    if hax > 7:
        _, leftDist = rc_utils.get_lidar_closest_point(scan, LEFT_WINDOW)
        _, rightDist = rc_utils.get_lidar_closest_point(scan, RIGHT_WINDOW)
        _, left45Dist = rc_utils.get_lidar_closest_point(scan, LEFT45_WINDOW)
        _, right45Dist = rc_utils.get_lidar_closest_point(scan, RIGHT45_WINDOW)

    if left45Dist > right45Dist and hax > 7:
        angle = rc_utils.clamp(right45Dist - left45Dist + 5, -1, 0)
    elif right45Dist > left45Dist and hax > 7:
        angle = rc_utils.clamp(right45Dist - left45Dist - 5, 0, 1)
    elif leftDist > rightDist and hax > 7:
        angle = rc_utils.clamp(rightDist - leftDist + 1, -1, 0)
    elif rightDist > leftDist and hax > 7:
        angle = rc_utils.clamp(rightDist - leftDist - 1, 0, 1)

    if hax > 7:
        speed = rc_utils.clamp(1.25 - angle, -1, 1)

def searchForMarkers():
    global speed
    global angle

    rc.drive.set_max_speed(0.25)
    speed = -0.5
    angle = 0

def start():
    rc.drive.stop()

def update():
    global speed
    global angle
    global priorityColor
    global switchTimer
    global switchSide
    global laneColor
    global wasFollowingWalls
    global curPart

    if curPart == Part.lineFollowing:
        followLine()
    elif curPart == Part.laneFollowing:
        followLane()
    elif curPart == Part.laneSwitching:
        switchLane()
    elif curPart == Part.coneSlaloming:
        slalomCones()
    elif curPart == Part.wallFollowing:
        followWalls()
    elif curPart == Part.searching:
        searchForMarkers()

    image = rc.camera.get_color_image()
    depthImg = rc.camera.get_depth_image()
    markers = rc_utils.get_ar_markers(image)

    for marker in markers:
        marker.detect_colors(image, potentialColors)
    
        if marker.get_id() == 0 and marker.get_orientation() == rc_utils.Orientation.LEFT:
            if priorityColor is None:
                priorityColor = marker.get_color()
            curPart = Part.lineFollowing
        elif marker.get_id() == 1 and marker.get_orientation() == rc_utils.Orientation.UP:
            if depthImg[marker.get_corners()[0][0]][marker.get_corners()[0][1]] < 400:
                laneColor = marker.get_color()
                curPart = Part.laneFollowing
        elif marker.get_id() == 199:
            if depthImg[marker.get_corners()[0][0]][marker.get_corners()[0][1]] < 200 and depthImg[marker.get_corners()[0][0]][marker.get_corners()[0][1]] > 1:
                if not wasFollowingWalls:
                    switchTimer = 0
                    switchSide = marker.get_orientation()
                    curPart = Part.laneSwitching
                else:
                    if depthImg[marker.get_corners()[0][0]][marker.get_corners()[0][1]] < 80:
                        switchTimer = 0
                        switchSide = marker.get_orientation()
                        curPart = Part.laneSwitching
        elif marker.get_id() == 2:
            if depthImg[marker.get_corners()[0][0]][marker.get_corners()[0][1]] < 220 and depthImg[marker.get_corners()[0][0]][marker.get_corners()[0][1]] > 1:
                curPart = Part.coneSlaloming
        elif marker.get_id() == 3 and marker.get_orientation() == rc_utils.Orientation.UP:
            if depthImg[marker.get_corners()[0][0]][marker.get_corners()[0][1]] < 90:
                curPart = Part.wallFollowing

    rc.drive.set_speed_angle(speed, angle)

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
