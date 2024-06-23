"""
Copyright MIT and Harvey Mudd College
MIT License
Fall 2020

Final Challenge - Grand Prix
"""

########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv
import numpy as np
import enum

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()
speed = 0.0
angle = 0.0
contour_center = None
contour_area = 0
color_image = None
cropped_image = None
depth_image = None
ar_markers = None
flag = False
marker = None

PURPLE = ((130, 150, 150), (140, 255, 255))
ORANGE = ((10, 150, 150), (20, 255, 255))
RED = ((170, 150, 150), (10, 255, 255))
GREEN = ((55, 150, 150), (65, 255, 255))
BLUE = ((100, 125, 125), (115, 255, 255))

potentialColors = [
    ((130, 150, 150), (140, 255, 255), "purple"),
    ((10, 50, 50), (20, 255, 255), "orange"),
    ((170, 50, 50), (10, 255, 255), "red"),
    ((40, 50, 50), (80, 255, 255), "green"),
    ((100, 150, 50), (110, 255, 255), "blue")
]

CROP_FLOOR = ((360, 0), (rc.camera.get_height(), rc.camera.get_width()))
LEFT45_WINDOW = (305, 325)
RIGHT45_WINDOW = (35, 55)
TURN_ANGLE = 4.9

CAMERA_HEIGHT = rc.camera.get_height()
CAMERA_WIDTH = rc.camera.get_width()
CAMERA_HEIGHTd2 = CAMERA_HEIGHT >> 1
CAMERA_WIDTHd2 = CAMERA_WIDTH >> 1

CROP_FLOOR_LINE = ((300, 0), (CAMERA_HEIGHT, CAMERA_WIDTH))
CROP_RIGHT = ((300, CAMERA_WIDTHd2), (CAMERA_HEIGHT, CAMERA_WIDTH))
CROP_LEFT = ((300, 0), (CAMERA_HEIGHT, CAMERA_WIDTHd2))
CROP_RIGHT_LANE_ACCEL = ((0, CAMERA_WIDTHd2), (CAMERA_HEIGHT, CAMERA_WIDTH))
CROP_LEFT_LANE_ACCEL = ((0, 0), (CAMERA_HEIGHT, CAMERA_WIDTHd2))

class State(enum.IntEnum):
    LineFollowing = 0
    WallFollowing = 1
    LaneFollowing = 2
    ArMarkers = 3
    Elevator = 4
    ConeSlaloming = 5
    TrainEvading = 6
    TileAvoiding = 7
    LaneFollowingAccelerated = 9

class Mode(enum.IntEnum):
    Normal = 0
    Left = 1
    Right = 2

cur_state = State.LineFollowing
cur_mode = Mode.Normal

laneColor = None
curLaneColor = None

changeLaneCoutner = 0

hax1 = 0
#_angle = 0

firstTurn = False

trainDist = -1
trainCounter = 0
TWENTY_PERCENT = 1/5
trainTimer = 0

turnDirMarkers = None # False -> left, True -> right
turnCounterMarkers = 0
dontFollowWalls = True
hasBegunTurning = False
isEvadingTrain = False

trainLeft = True
trainCenter = True
trainRight = True
brakingNews = False

isEvadingTrainWhenBraking = None
prevClosestDist = 0

prevTrainCenter = True
dontEvadeTrainCarIsTooFarAway = False
firstTimeBraking1 = True
firstTimeBraking2 = True




prevTrainCenter = False
evadeTrain = False
evadeTrainTimer = 0


# Cone Slaloming
class ConeState(enum.IntEnum):
    search  = 0
    approach = 1
    LiDAR = 2
    turn = 3

coneState = ConeState.search

OFFSET = 150
DIST = 80

curConeState = ConeState.search
curConeColor = None
coneDist = 9999

CONE_BLUE = ((100, 175, 200), (130, 255, 255))
CONE_RED = ((165, 180, 150),(179, 255, 255))
########################################################################################
# Functions
########################################################################################

# Done
def lineFollowing():
    global speed
    global angle

    if contour_center is not None:
        angle = rc_utils.remap_range(contour_center[1], 0, rc.camera.get_width(), -1, 1)

    speed = 1
# Done
def wallFollowing():
    global speed
    global angle
    global LEFT45_WINDOW
    global RIGHT45_WINDOW

    scan = rc.lidar.get_samples()
    _, left45Dist = rc_utils.get_lidar_closest_point(scan, LEFT45_WINDOW)
    _, right45Dist = rc_utils.get_lidar_closest_point(scan, RIGHT45_WINDOW)

    if left45Dist > right45Dist:
        angle = rc_utils.clamp(right45Dist - left45Dist + 5, -1, 0)
    else:
        angle = rc_utils.clamp(right45Dist - left45Dist - 5, 0, 1)

    speed = rc_utils.clamp(1.25 - angle, -1, 1)
# Done
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
# Done
def detectContours2(colorImg):
    global laneColor

    if colorImg is None:
        print("**COLOR IMAGE IS NONE**")
    else:
        contour = rc_utils.get_largest_contour(rc_utils.find_contours(colorImg, BLUE[0], BLUE[1]))
        return rc_utils.get_contour_center(contour)
# Fix
def laneFollowing():
    global speed
    global angle

    global speed
    global angle
    global laneColor
    global curLaneColor
    global changeLaneCoutner
    global hax1
    #global _angle
    global firstTurn

    speed = 1

    image = rc.camera.get_color_image()

    colorImg1 = rc_utils.crop(image, CROP_RIGHT[0], CROP_RIGHT[1])
    colorImg2 = rc_utils.crop(image, CROP_LEFT[0], CROP_LEFT[1])
    colorImg3 = rc_utils.crop(image, CROP_FLOOR_LINE[0], CROP_FLOOR_LINE[1])
    (contourCenterP1, contourCenterO1, contourPBiggerThanContourO1) = detectContours(colorImg1) #right
    (contourCenterP2, contourCenterO2, contourPBiggerThanContourO2) = detectContours(colorImg2) #left

    if curLaneColor == None:
        curLaneColor = laneColor

    # If the lane that the car is following (or should follow) is purple
    if curLaneColor == "purple":
        # If there are 2 purple lines detected
        if contourCenterP1 is not None and contourCenterP2 is not None and not (None in contourCenterP1) and not (None in contourCenterP2):
            # The angles for the 2 images (The color image is cut in half)
            angle1 = rc_utils.remap_range(contourCenterP1[1], 0, CAMERA_WIDTHd2, -1, 1)
            angle2 = rc_utils.remap_range(contourCenterP2[1], 0, CAMERA_WIDTHd2, -1, 1)

            rc_utils.draw_circle(colorImg3, (contourCenterP1[0], contourCenterP1[1]+(CAMERA_WIDTHd2)))
            rc_utils.draw_circle(colorImg3, contourCenterP2)

            rc.display.show_color_image(colorImg3)

            angle = rc_utils.clamp(angle1 + angle2, -1, 1)
            #speed = rc_utils.clamp(1 - 0.5*abs(angle), -1, 1)

        # If 2 purple lines are not detected
        elif (contourCenterP1 is None or (None in contourCenterP1)) or (contourCenterP2 is None or (None in contourCenterP2)):
            #print("Less than 2 purple lines detected")
            # If atleast 1 orange line is detected, make the car follow the oragne lane
            if (contourCenterO1 is not None and not (None in contourCenterO1)) or (contourCenterO2 is not None and not (None in contourCenterO2)):
                changeLaneCoutner += 1
                curLaneColor = "orange"
            
    # If the lane that the car is following (or should follow) is orange
    elif curLaneColor == "orange":
        # If there are 2 orange lines detected
        if contourCenterO1 is not None and contourCenterO2 is not None and not (None in contourCenterO1) and not (None in contourCenterO2):
            angle1 = rc_utils.remap_range(contourCenterO1[1], 0, CAMERA_WIDTHd2, -1, 1)
            angle2 = rc_utils.remap_range(contourCenterO2[1], 0, CAMERA_WIDTHd2, -1, 1)

            rc_utils.draw_circle(colorImg3, (contourCenterO1[0], contourCenterO1[1]+(CAMERA_WIDTHd2)))
            rc_utils.draw_circle(colorImg3, contourCenterO2)

            rc.display.show_color_image(colorImg3)
            
            angle = rc_utils.clamp(angle1 + angle2, -1, 1)
            #speed = rc_utils.clamp(1 - 0.5*abs(angle), -1, 1)
    
        # If 2 orange lines are not detected
        elif (contourCenterO1 is None or (None in contourCenterO1)) or (contourCenterO2 is None or (None in contourCenterO2)):
            #print("Less than 2 orange lines detected")
            # If atleast 1 purple line is detected, make the car follow the purple lane
            if (contourCenterP1 is not None and not (None in contourCenterP1)) or (contourCenterP2 is not None and not (None in contourCenterP2)):
                changeLaneCoutner += 1
                curLaneColor = "purple"

    if curLaneColor != laneColor:
        if changeLaneCoutner == 1:
            #_angle = 0.9
            if curLaneColor == "purple":
                print("1st Sharp turn!")
                print("Cur Lane Color -> purple")
                contourCenterP1 = (CAMERA_HEIGHT - 200, CAMERA_WIDTH - 50)

                # Recalculate the angle based on the new contour center
                angle1 = rc_utils.remap_range(contourCenterP1[1], 0, CAMERA_WIDTHd2, -1, 1)
                angle2 = rc_utils.remap_range(contourCenterP2[1], 0, CAMERA_WIDTHd2, -1, 1)
            
                angle = rc_utils.clamp(angle1 + angle2, -1, 1)
                #rc_utils.draw_circle(colorImg3, (contourCenterP1[0], contourCenterP1[1]+(CAMERA_WIDTHd2)))
                #rc_utils.draw_circle(colorImg3, contourCenterP2)
            else:
                print("1st Sharp turn!")
                print("Cur Lane Color -> orange")
                contourCenterO1 = (CAMERA_HEIGHT - 200, CAMERA_WIDTH - 50)

                # Recalculate the angle based on the new contour center
                angle1 = rc_utils.remap_range(contourCenterO1[1], 0, CAMERA_WIDTHd2, -1, 1)
                angle2 = rc_utils.remap_range(contourCenterO2[1], 0, CAMERA_WIDTHd2, -1, 1)
            
                angle = rc_utils.clamp(angle1 + angle2, -1, 1)
                #rc_utils.draw_circle(colorImg3, (contourCenterO1[0], contourCenterO1[1]+(CAMERA_WIDTHd2)))
                #rc_utils.draw_circle(colorImg3, contourCenterO2)
        if changeLaneCoutner != 1:
            #_angle = -0.9
            if curLaneColor == "purple":
                print("2nd Sharp turn!")
                print("Cur Lane Color -> purple")
                contourCenterP2 = (CAMERA_HEIGHT - 200, 50)

                # Recalculate the angle based on the new contour center
                angle1 = rc_utils.remap_range(contourCenterP1[1], 0, CAMERA_WIDTHd2, -1, 1)
                angle2 = rc_utils.remap_range(contourCenterP2[1], 0, CAMERA_WIDTHd2, -1, 1)
            
                angle = rc_utils.clamp(angle1 + angle2, -1, 1)
                #rc_utils.draw_circle(colorImg3, (contourCenterP1[0], contourCenterP1[1]+(CAMERA_WIDTHd2)))
                #rc_utils.draw_circle(colorImg3, contourCenterP2)
            else:
                print("2nd Sharp turn!")
                print("Cur Lane Color -> orange")
                contourCenterO2 = (CAMERA_HEIGHT - 200, 50)

                # Recalculate the angle based on the new contour center
                angle1 = rc_utils.remap_range(contourCenterO1[1], 0, CAMERA_WIDTHd2, -1, 1)
                angle2 = rc_utils.remap_range(contourCenterO2[1], 0, CAMERA_WIDTHd2, -1, 1)
            
                angle = rc_utils.clamp(angle1 + angle2, -1, 1)
                #rc_utils.draw_circle(colorImg3, (contourCenterO1[0], contourCenterO1[1]+(CAMERA_WIDTHd2)))
                #rc_utils.draw_circle(colorImg3, contourCenterO2)

    #if _angle:
    #    hax1 += rc.get_delta_time()
    #    if not firstTurn:
    #        if hax1 < 0.25:
    #            pass
    #        elif hax1 < 2.2:
    #            angle = _angle
    #        else:
    #            hax1 = 0
    #            _angle = 0
    #            firstTurn = True
    #    else:
    #        if hax1 < 1.2:
    #            pass
    #        elif hax1 < 2.05:
    #            angle = _angle
    #        else:
    #            hax1 = 0
    #            _angle = 0

def laneFollowing2():
    global speed
    global angle

    image = rc.camera.get_color_image()

    if image is not None:
        contoursP = rc_utils.find_contours(image, PURPLE[0], PURPLE[1])
        contoursO = rc_utils.find_contours(image, ORANGE[0], ORANGE[1])
# Done
def arMarkers():
    global speed
    global angle
    global turnDirMarkers
    global turnCounterMarkers
    global dontFollowWalls
    global hasBegunTurning

    if turnDirMarkers != None:
        if hasBegunTurning == False:
            turnCounterMarkers = -1
    elif dontFollowWalls == False:
        scan = rc.lidar.get_samples()
        (_, left45Dist) = rc_utils.get_lidar_closest_point(scan, LEFT45_WINDOW)
        (_, right45Dist) = rc_utils.get_lidar_closest_point(scan, RIGHT45_WINDOW)

        if left45Dist > right45Dist:
            angle = rc_utils.clamp(right45Dist - left45Dist + 5, -1, 0)
        else:
            angle = rc_utils.clamp(right45Dist - left45Dist - 5, 0, 1)

        speed = rc_utils.clamp(1.5 - angle, -1, 1)

    if turnCounterMarkers == -1 or turnCounterMarkers > 0:
        hasBegunTurning = True
        if turnCounterMarkers == -1:
            turnCounterMarkers = 0
        
        turnCounterMarkers += rc.get_delta_time()
        if turnCounterMarkers < 0.5:
            if turnDirMarkers:
                angle = 0.75
            else:
                angle = -0.75
        else:
            turnDirMarkers = None
            turnCounterMarkers = 0
            hasBegunTurning = False
# Done
def elevator():
    global speed
    global angle
    global ar_markers

    if ar_markers:
        marker = ar_markers[0]
        corners = marker.get_corners()
        ar_center = ((corners[0][0] + corners[2][0]) >> 1, (corners[0][1] + corners[2][1]) >> 1)
        if depth_image[ar_center[0]][ar_center[1]] < 300:
            ar_markers[0].detect_colors(color_image, potentialColors)
            id = marker.get_id()
            ar_color = marker.get_color()
            if id == 3:
                if ar_color == "blue" or ar_color == "orange":
                    # Move forward
                    speed = 1
                    rc.drive.set_max_speed(0.8)
                    angle = rc_utils.remap_range(ar_center[1], 0, CAMERA_WIDTH, -1, 1)
                elif ar_color == "red":
                    # Stop
                    speed = 0
                    angle = rc_utils.remap_range(ar_center[1], 0, CAMERA_WIDTH, -1, 1)
# Done
def offset(x):
    return rc_utils.clamp(275-x, 0, CAMERA_WIDTH)

def findRedCone(image):
    if image is None:
        print("*** IMAGE IS NONE ***")
        return None
    else:
        return rc_utils.get_largest_contour(rc_utils.find_contours(image, CONE_RED[0], CONE_RED[1]), 800)

def findBlueCone(image):
    if image is None:
        print("*** IMAGE IS NONE ***")
        return None
    else:
        return rc_utils.get_largest_contour(rc_utils.find_contours(image, CONE_BLUE[0], CONE_BLUE[1]), 800)

# ~Done: Passes very close to some cones
# Sometimes it doesn't change from LiDAR to turn
def coneSlaloming():
    global angle, speed
    global curConeState
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
    if curConeState == ConeState.search or curConeState == ConeState.turn:
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
        if coneDist > DIST:# and (curConeState == ConeState.turn or curConeState == ConeState.search):
            curConeState = ConeState.approach
    
    if curConeState == ConeState.approach:
        if coneCenter is not None:
            rc_utils.draw_circle(image, coneCenter)
            if curConeColor == False: # red
                if coneCenter[1] + OFFSET >= CAMERA_WIDTH:
                    rc_utils.draw_circle(image, (coneCenter[0], CAMERA_WIDTH-1))
                else:
                    rc_utils.draw_circle(image, (coneCenter[0], coneCenter[1] + OFFSET))
                angle = rc_utils.remap_range(coneCenter[1] + offset(depthImage[coneCenter[0]][coneCenter[1]]), 0, CAMERA_WIDTH, -1, 1, True)
                
                if angle < 0.9 and angle > -0.9:
                    # Its time to switch to the LiDAR to track the cone
                    # The car will move towards the cone and when the cone is at the correct position the car will turn
                    curConeState = ConeState.LiDAR
            elif curConeColor == True: # blue
                if coneCenter[1] - OFFSET < 0:
                    rc_utils.draw_circle(image, (coneCenter[0], 0))
                else:
                    rc_utils.draw_circle(image, (coneCenter[0], coneCenter[1] - OFFSET))
                angle = rc_utils.remap_range(coneCenter[1] - offset(depthImage[coneCenter[0]][coneCenter[1]]), 0, CAMERA_WIDTH, -1, 1, True)
               
                if angle < 0.9 and angle > -0.9:
                    # Its time to switch to the LiDAR to track the cone
                    # The car will move towards the cone and when the cone is at the correct position the car will turn
                    curConeState = ConeState.LiDAR

    elif curConeState == ConeState.LiDAR:
        scan = rc.lidar.get_samples()

        coneAngle, coneDist = rc_utils.get_lidar_closest_point(scan)
        print("Cone Dist (LiDAR):", coneDist)
        print("Cone Angle (LiDAR):", coneAngle)
        if coneDist < 100: # The LiDAR can see the cone
            if coneAngle < 70 and coneAngle > 55: # The cone is at the correct position
                curConeState = ConeState.turn
                # Turn correctly based on the color of the cone
            elif coneAngle < 310 and coneAngle > 285:
                curConeState = ConeState.turn
                # Turn correctly based on the color of the cone
    elif curConeState == ConeState.turn:
        # Look for cones
        # If there are none, keep turning left/right
        # The cone must be clone enough
        #angle = 0.75
        if coneCenter is not None and not (None in coneCenter) and depthImage[coneCenter[0]][coneCenter[1]] < 120:
            curConeState = ConeState.approach
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
    #rc.drive.set_speed_angle(speed, angle)

    print("State:", curConeState)
    print("Cur Cone Color:", curConeColor)
# Too complicated, Not Working
def trainEvading():
    global speed
    global angle
    global trainDist
    global trainCounter
    global trainTimer
    global isEvadingTrain
    global trainLeft
    global trainCenter
    global trainRight
    global cur_state
    global brakingNews
    global isEvadingTrainWhenBraking
    global prevClosestDist
    global prevTrainCenter
    global dontEvadeTrainCarIsTooFarAway
    global firstTimeBraking1
    global firstTimeBraking2

    rc.drive.set_max_speed(0.25)

    angle = 0
    speed = 0

    scan = rc.lidar.get_samples()

    _, left10Dist = rc_utils.get_lidar_closest_point(scan, (338, 341))
    _, frontDist = rc_utils.get_lidar_closest_point(scan, (359, 1))
    _, right10Dist = rc_utils.get_lidar_closest_point(scan, (18, 22))
    _, left125Dist = rc_utils.get_lidar_closest_point(scan, (334, 336))
    _, right125Dist = rc_utils.get_lidar_closest_point(scan, (24, 26))

    trainDist = 140

    if left10Dist < trainDist:
        trainLeft = True
    else:
        trainLeft = False
    
    if frontDist < trainDist:
        trainCenter = True
    else:
        trainCenter = False
    
    if right10Dist < trainDist:
        trainRight = True
    else:
        trainRight = False

    if left125Dist < trainDist:
        trainFarLeft = True
    else:
        trainFarLeft = False

    if right125Dist < trainDist:
        trainFarRight = True
    else:
        trainFarRight = False

    _, closestPointLeft = rc_utils.get_lidar_closest_point(scan, (270, 0))
    _, closestPointRight = rc_utils.get_lidar_closest_point(scan, (0, 90))

    print("Closest pointL:", closestPointLeft)
    print("Closest pointR:", closestPointRight)

    # This is not a train (0)
    if trainCounter == 0:
        if trainLeft == True and trainCenter == False and trainRight == False:
            if isEvadingTrain == False:
                isEvadingTrain = True
                trainTimer = 0
                brakingNews = False
        elif trainLeft == False and trainCenter == False and trainRight == False:
            if isEvadingTrain == False:
                isEvadingTrain = True
                trainTimer = 0
                brakingNews = False
        else:
            brakingNews = True
    elif trainCounter == 1:
        #if trainLeft == True and trainCenter == False and trainRight == False and trainFarRight == False:
        #    if isEvadingTrain == False and closestPointRight > 100:
        if prevTrainCenter == True and trainCenter == False and dontEvadeTrainCarIsTooFarAway == False and trainFarRight == False:
            isEvadingTrain = True
            trainTimer = 0
            brakingNews = False
        elif trainLeft == False and trainCenter == False and trainRight == False and isEvadingTrain == True:
            pass
        elif isEvadingTrain == True:
            pass
        else:
            brakingNews = True
    elif trainCounter == 2:
        #if trainLeft == False and trainCenter == False and trainRight == True and trainFarLeft == False:
        #    if isEvadingTrain == False and closestPointLeft > 100:
        if prevTrainCenter == True and trainCenter == False and dontEvadeTrainCarIsTooFarAway == False and trainFarLeft == False:
            isEvadingTrain = True
            trainTimer = 0
            brakingNews = False
        elif trainLeft == False and trainCenter == False and trainRight == False and isEvadingTrain == True:
            pass
        elif isEvadingTrain == True:
            pass
        else:
            brakingNews = True
    elif trainCounter == 3:
        #if trainLeft == True and trainCenter == False and trainRight == False and trainFarRight == False:
        #    if isEvadingTrain == False and closestPointRight > 100:
        if prevTrainCenter == True and trainCenter == False and dontEvadeTrainCarIsTooFarAway == False and trainFarRight == False:
            isEvadingTrain = True
            trainTimer = 0
            brakingNews = False
        elif trainLeft == False and trainCenter == False and trainRight == False and isEvadingTrain == True:
            pass
        elif isEvadingTrain == True:
            pass
        else:
            brakingNews = True
    else:
        cur_state = State.LineFollowing
        lineFollowing()

    #_, leftDist = rc_utils.get_lidar_closest_point(scan, (265, 275))
    #_, rightDist = rc_utils.get_lidar_closest_point(scan, (85, 95))

    #if leftDist > rightDist:
    #    angle = rc_utils.clamp(rightDist - leftDist + 1, -1, 0)
    #elif rightDist > leftDist:
    #    angle = rc_utils.clamp(rightDist - leftDist - 1, 0, 1)


    if isEvadingTrain:
        trainTimer += rc.get_delta_time()
        angle = 0

        if trainTimer < 2:
            print("Evading Train")
            if trainCounter != 0:
                rc.drive.set_max_speed(0.4)
            else:
                rc.drive.set_max_speed(0.25)
            speed = 1
        else:
            trainCounter += 1
            isEvadingTrain = False
            speed = -0.5
            rc.drive.set_max_speed(0.25)

    if brakingNews:
        rc.drive.set_max_speed(0.6)
        # The following code is wrong
        if isEvadingTrain == True and isEvadingTrainWhenBraking == None:
            isEvadingTrainWhenBraking = True
        else:
            isEvadingTrainWhenBraking = False
        
        if isEvadingTrainWhenBraking == False and isEvadingTrain == True:
            #isEvadingTrain == False
            brakingNews = False
            isEvadingTrainWhenBraking = None

        _, closestPoint = rc_utils.get_lidar_closest_point(scan, (270, 90))

        if closestPoint < 40:
            dontEvadeTrainCarIsTooFarAway = False
            #if firstTimeBraking2 == True:
            #    firstTimeBraking2 = False
            rc.drive.set_max_speed(1)
            speed = -1
            #else:
            #    rc.drive.set_max_speed(0.6)
            #    speed = -0.5
        if closestPoint < 50:
            dontEvadeTrainCarIsTooFarAway = False

            #if firstTimeBraking1 == True and firstTimeBraking2 == False:
            #    firstTimeBraking1 = False

            #if firstTimeBraking2 == True:
            #    firstTimeBraking2 = False
            rc.drive.set_max_speed(0.6)
            speed = -1
            #else:
            #    rc.drive.set_max_speed(0.4)
            #    speed = 0
        elif closestPoint < 60:
            dontEvadeTrainCarIsTooFarAway = False
            speed = 0
        elif closestPoint < 120:
            dontEvadeTrainCarIsTooFarAway = True
            rc.drive.set_max_speed(0.25)
            speed = 0.25
        else:
            dontEvadeTrainCarIsTooFarAway = False
            brakingNews = False
            isEvadingTrainWhenBraking = None
            #firstTimeBraking1 = True
            #firstTimeBraking2 = True
    #else:
    #    firstTimeBraking1 = True
    #    firstTimeBraking2 = True

    prevTrainCenter = trainCenter

    print("Train Dist:", trainDist)
    print("Train counter:", trainCounter)
    print("scan[-20]:", left10Dist)
    print("scan[0]:", frontDist)
    print("scan[20]:", right10Dist)
    print("FarLeft:", trainFarLeft)
    print("Left:", trainLeft)
    print("Center:", trainCenter)
    print("Right:", trainRight)
    print("FarRight:", trainFarRight)
    print("isEvadingTrain:", isEvadingTrain)
    print("brakingNews:", brakingNews)
    print("\nprevTrainCenter:", prevTrainCenter)
    print("traincenter:", trainCenter)
    print("\n")
    print("DontEvadeTrain:", dontEvadeTrainCarIsTooFarAway)
# Fix: don't start if too far away from the train
# Almost no time gain or ~7-10 seconds slower than line following
def trainEvading2():
    global speed
    global prevTrainCenter
    global evadeTrain
    global evadeTrainTimer
    global trainCounter

    scan = rc.lidar.get_samples()

    _, centerDist = rc_utils.get_lidar_closest_point(scan, (359, 1))
    _, closestPoint = rc_utils.get_lidar_closest_point(scan, (270, 90))
    brakingNews = False

    if centerDist < 140:
        trainCenter = True
    else:
        trainCenter = False

    if prevTrainCenter == True and trainCenter == False and closestPoint < 80:
        if evadeTrain == False:
            evadeTrain = True
            evadeTrainTimer = 0
    elif evadeTrain == False:
        brakingNews = True
    
    if evadeTrain == True:
        evadeTrainTimer += rc.get_delta_time()

        if evadeTrainTimer < 0.5:
            rc.drive.set_max_speed(0.6)
            speed = 1
        elif evadeTrainTimer < 1.5:
            rc.drive.set_max_speed(0.4)
            speed = 1
        else:
            rc.drive.set_max_speed(0.25)
            speed = 0.5
            trainCounter += 1
            evadeTrain = False
    
    if brakingNews == True:
        if closestPoint < 50:
            speed = -1
        elif closestPoint < 70:
            speed = 0
        else:
            speed = 0.25

    prevTrainCenter = trainCenter

    print("trainCenter:", trainCenter)
    print("EvadeTrain:", evadeTrain)
    print("brakingNews:", brakingNews)
    print("closestPoint:", closestPoint)

def tileAvoiding():
    global speed
    global angle
    
    print("TILE AVOIDING")
    speed = angle = 0

def laneFollowingAccelerated():
    global speed
    global angle

    """speed = 1

    image = rc.camera.get_color_image()

    colorImg1 = rc_utils.crop(image, CROP_RIGHT_LANE_ACCEL[0], CROP_RIGHT_LANE_ACCEL[1])
    colorImg2 = rc_utils.crop(image, CROP_LEFT_LANE_ACCEL[0], CROP_LEFT_LANE_ACCEL[1])
    contourCenter1 = detectContours2(colorImg1) #right
    contourCenter2 = detectContours2(colorImg2) #left

        # If there are 2 lines detected
    if contourCenter1 is not None and contourCenter2 is not None and not (None in contourCenter1) and not (None in contourCenter2):
        print("Found 2 contours")
        print("ContourCenter1:", contourCenter1)
        print("ContourCenter2:", contourCenter2)
        # The angles for the 2 images (The color image is cut in half)
        angle1 = rc_utils.remap_range(contourCenter1[1], 0, CAMERA_WIDTHd2, -1, 1) * 0.4
        angle2 = rc_utils.remap_range(contourCenter2[1], 0, CAMERA_WIDTHd2, -1, 1) * 0.6

        print("Angle1:", angle1)
        print("Angle2:", angle2)

        rc_utils.draw_circle(image, (contourCenter1[0], contourCenter1[1]+(CAMERA_WIDTHd2)))
        rc_utils.draw_circle(image, contourCenter2)

        rc.display.show_color_image(image)

        angle = rc_utils.clamp(angle1 + angle2, -1, 1)
        #speed = rc_utils.clamp(1 - 0.5*abs(angle), -1, 1)"""
    

    global timer

    color_image_new = rc.camera.get_color_image()
    #cropped_image_new = rc_utils.crop(color_image_new, CROP_FLOOR_LANE_ACCEL[0], CROP_FLOOR_LANE_ACCEL[1])
    cropped_image_new = color_image_new
    speed = 1
    contours = rc_utils.find_contours(cropped_image_new, BLUE[0], BLUE[1])
    contour_blue = rc_utils.get_largest_contour(contours, 1200)
    if contour_blue is not None and not (None in contour_blue):
        contour_center_blue = rc_utils.get_contour_center(contour_blue)
        angle1 = rc_utils.remap_range(contour_center_blue[1], 0, CAMERA_WIDTH, -1, 1)
        contours = tuple([x for x in contours if x.all() != contour_blue.all()]) #contours.remove(contour_blue)
        rc_utils.draw_circle(cropped_image_new, contour_center_blue)

        contour_blue_second = rc_utils.get_largest_contour(contours, 1200)
        if contour_blue_second is not None and not (None in contour_blue_second):
            contour_center_blue_second = rc_utils.get_contour_center(contour_blue_second)
            angle2 = rc_utils.remap_range(contour_center_blue_second[1], 0, CAMERA_WIDTH, -1, 1)
            rc_utils.draw_circle(cropped_image_new, contour_center_blue_second)
            #mid_contour_center = ((contour_center_blue[0] + contour_center_blue_second[0])>>1, (contour_center_blue[1] + contour_center_blue_second[1])>>1)
            #rc_utils.draw_circle(cropped_image_new, mid_contour_center)
            #angle = rc_utils.remap_range(mid_contour_center[1], 0, CAMERA_WIDTH, -1, 1)
            if timer < 12:
                angle = angle1 * 0.2 + angle2 * 0.8
            else:
                angle = angle1 * 0.6 + angle2 * 0.4
        else:
            #angle = rc_utils.remap_range(contour_center_blue[1], 0, CAMERA_WIDTH, -1, 1)
            if timer < 12:
                angle = -angle1
            else:
                angle = angle1

    angle = rc_utils.clamp(angle, -1, 1)
    rc.display.show_color_image(cropped_image_new)
# Done
def update_contour():
    """
    Finds contours in the current color image and uses them to update contour_center
    and contour_area
    """
    global contour_center
    global contour_area
    global cropped_image

    if cropped_image is None:
        contour_center = None
        contour_area = 0
    else:
        contour = rc_utils.get_largest_contour(rc_utils.find_contours(cropped_image, GREEN[0], GREEN[1]))
        if contour is not None:
                contour_center = rc_utils.get_contour_center(contour)
                contour_area = rc_utils.get_contour_area(contour)

                rc_utils.draw_contour(cropped_image, contour)
                rc_utils.draw_circle(cropped_image, contour_center)
        else:
            contour_center = None
            contour_area = 0
# Done
def start():
    global speed
    global angle
    global cur_state
    global cur_mode
    global color_image
    global cropped_image
    global depth_image
    global corner
    global timer
    global marker
    global flag

    cur_state = State.LineFollowing
    cur_mode = Mode.Normal
    speed = 0
    angle = 0
    color_image = None
    cropped_image = None
    depth_image = None
    marker = None
    timer = 0
    flag = False
    corner = None
    rc.drive.stop()

    print(">> Final Challenge - Grand Prix")
# Done
def update():
    global speed
    global angle
    global cur_state
    global color_image
    global cropped_image
    global depth_image
    global ar_markers
    global marker
    global corner
    global timer
    global flag
    global laneColor
    global turnDirMarkers
    global dontFollowWalls
    global trainCounter

    color_image = rc.camera.get_color_image()
    cropped_image = rc_utils.crop(color_image, CROP_FLOOR[0], CROP_FLOOR[1])
    depth_image = rc.camera.get_depth_image()
    ar_markers = rc_utils.get_ar_markers(color_image)
    dt = rc.get_delta_time()

    update_contour()
    if cur_state == State.LineFollowing:
        if dt < 0.033:
            rc.drive.set_max_speed(0.275) #3
        elif dt < 0.085:
            rc.drive.set_max_speed(0.25) #3
        else:
            rc.drive.set_max_speed(0.225) #3

        if flag == False:
            if dt < 0.033:
                rc.drive.set_max_speed(0.6) #1
            elif dt < 0.065:
                rc.drive.set_max_speed(0.55) #1
            elif dt < 0.08:
                rc.drive.set_max_speed(0.5) #1
            else:
                rc.drive.set_max_speed(0.4) #1
        lineFollowing()
        if ar_markers:
            marker = ar_markers[0]
            corners = marker.get_corners()
            ar_center = ((corners[0][0] + corners[2][0]) >> 1, (corners[0][1] + corners[2][1]) >> 1)
            if depth_image[ar_center[0]][ar_center[1]] < 80:
                id = marker.get_id()
                if id == 0:
                    timer = 0
                    if flag:
                        cur_state = State.ArMarkers
                    else:
                        cur_state = State.WallFollowing
                        flag = True              
                elif id == 1:
                    cur_state = State.LaneFollowing
                    timer = 0
                elif id == 3:
                    cur_state = State.Elevator
                elif id == 4:
                    cur_state = State.ConeSlaloming
                elif id == 5 and trainCounter == 0:
                    cur_state = State.TrainEvading
                    timer = 0
                elif id == 6:
                    cur_state = State.TileAvoiding
                elif id == 8:
                    cur_state = State.LaneFollowingAccelerated
                    timer = 0
            elif depth_image[ar_center[0]][ar_center[1]] < 300:
                id = marker.get_id()
                if id == 3:
                    cur_state = State.Elevator
    # Done
    elif cur_state == State.WallFollowing:
        if dt < 0.033:
            rc.drive.set_max_speed(0.28) #2
        elif dt < 0.08:
            rc.drive.set_max_speed(0.275) #2
        else:
            rc.drive.set_max_speed(0.25) #2
        
        timer += rc.get_delta_time()
        if timer > 1:
            if contour_center is None:
                wallFollowing()
            else:
                timer = 0
                cur_state = State.LineFollowing
        else:
            speed = 1
            angle = 0
    # Done
    elif cur_state == State.ArMarkers:
        rc.drive.set_max_speed(0.25)
        timer += rc.get_delta_time()

        speed = 1

        for marker in ar_markers:
            if marker.get_id() == 199:
                corners = marker.get_corners()
                ar_center = ((corners[0][0] + corners[2][0]) >> 1, (corners[0][1] + corners[2][1]) >> 1)
                if depth_image[ar_center[0]][ar_center[1]] < 80:
                    if marker.get_orientation() == rc_utils.Orientation.LEFT:
                        turnDirMarkers = False # False -> left, True -> right
                    else:
                        turnDirMarkers = True
                    dontFollowWalls = False
                else:
                    angle = rc_utils.remap_range(ar_center[1], 0, CAMERA_WIDTH, -1, 1)
                    dontFollowWalls = True
            elif marker.get_id() == 0:
                corners = marker.get_corners()
                ar_center = ((corners[0][0] + corners[2][0]) >> 1, (corners[0][1] + corners[2][1]) >> 1)
                angle = rc_utils.remap_range(ar_center[1], 0, CAMERA_WIDTH, -1, 1)
                dontFollowWalls = True
        arMarkers()

        if contour_center is not None and depth_image[contour_center[0]][contour_center[1]] < 550 and timer > 2:
            cur_state = State.LineFollowing
    # Temp Disabled - Balsis is doing this
    # TODO: Fix
    elif cur_state == State.LaneFollowing:
        rc.drive.set_max_speed(0.1)
        speed = 0.1
        #if ar_markers:
        #    ar_markers[0].detect_colors(color_image, potentialColors)
        #    laneColor = ar_markers[0].get_color()
        #timer += rc.get_delta_time()
        #if timer > 1:
        #    if contour_center is None:
        #        laneFollowing()
        #    elif timer > 4:
        #        timer = 0
        #        cur_state = State.LineFollowing
        #else:
        #    laneFollowing()
        cur_state = State.LineFollowing
        lineFollowing()
    # Done
    elif cur_state == State.Elevator:
        rc.drive.set_max_speed(0.25)
        elevator()
        timer += rc.get_delta_time()
        if timer > 7:
            if contour_center is not None:
                timer = 0
                cur_state = State.LineFollowing
    # Temp (hopefully) Disabled - Myrsini is doing this
    # Enabled - I did it
    elif cur_state == State.ConeSlaloming:
        #timer += rc.get_delta_time()
        #coneSlaloming()

        #if ar_markers:
        #    for marker in ar_markers:
        #        if marker.get_id() == 4:
        #            rc.drive.set_max_speed(0.15)
        #            speed = 0.1

        #if contour_center is not None and depth_image[contour_center[0]][contour_center[1]] < 550 and timer > 10:
        #    print("Line Dist:", depth_image[contour_center[0]][contour_center[1]])
        #    lineFollowing()
        #    cur_state = State.LineFollowing
        lineFollowing()
        #rc.drive.set_max_speed(0.15) This doesn't actually do anything - only when the car sees the AR Marker
        cur_state = State.LineFollowing
    # Small Fix - Active (very low time gain -5-4 seconds)
    elif cur_state == State.TrainEvading:
        rc.drive.set_max_speed(0.25)
        timer += rc.get_delta_time()

        ar_dist = 0

        if ar_markers:
            for marker in ar_markers:
                if marker.get_id() == 5 and trainCounter != 0:
                    cur_state = State.LineFollowing
            corners = ar_markers[0].get_corners()
            ar_center = ((corners[0][0] + corners[2][0]) >> 1, (corners[0][1] + corners[2][1]) >> 1)
            ar_dist = depth_image[ar_center[0]][ar_center[1]]
            angle = rc_utils.remap_range(ar_center[1], 0, CAMERA_WIDTH, -1, 1) - 0.05
        
        print("isLineVisible:", contour_center is not None)
        if contour_center is not None:
            print("lineDist:", depth_image[contour_center[0]][contour_center[1]])
        
        if contour_center is not None and depth_image[contour_center[0]][contour_center[1]] < 550 and timer > 5:
            cur_state = State.LineFollowing
            timer = 0
        elif ar_dist < 50:
            trainEvading2()
            angle = 0
    # Disabled - Not done yet
    elif cur_state == State.TileAvoiding:
        #rc.drive.set_max_speed(0.25)
        #tileAvoiding()
        lineFollowing()
        flag = None # To change max speed to 0.3
        cur_state = State.LineFollowing
    # Disabled - very low time gain 4-6 seconds
    elif cur_state == State.LaneFollowingAccelerated:
        if dt < 0.033:
            rc.drive.set_max_speed(0.5) #1
        elif dt < 0.065:
            rc.drive.set_max_speed(0.45) #1
        elif dt < 0.08:
            rc.drive.set_max_speed(0.4) #1
        else:
            rc.drive.set_max_speed(0.35) #1
        #laneFollowingAccelerated()
        lineFollowing()

    print(cur_state, timer)
    # Πίσω
    scan = rc.lidar.get_samples()
    closestPointAngle, closestPoint = rc_utils.get_lidar_closest_point(scan, (305, 55))
    if closestPoint < 20 and cur_state != State.Elevator:
        speed = -1
        if closestPointAngle > 270:
            angle = -1
        else:
            angle = 1
    rc.drive.set_speed_angle(speed, angle)

    print(rc.get_delta_time())

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()