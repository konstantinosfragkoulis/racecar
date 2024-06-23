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
speed:float = 0
angle:float = 0
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

CAMERA_HEIGHT = rc.camera.get_height()
CAMERA_WIDTH = rc.camera.get_width()

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

cur_state = State.LineFollowing

trainCounter = 0
trainTimer = 0

turnDirMarkers = None # False -> left, True -> right
turnCounterMarkers = 0
dontFollowWalls = True
hasBegunTurning = False

trainCenter = True
brakingNews = False

prevTrainCenter = True

prevTrainCenter = False
evadeTrain = False
evadeTrainTimer = 0

overrideMaxSpeed = False
coneSpeed = False

collisionDetection = False
########################################################################################
# Functions
########################################################################################

def lineFollowing():
    global speed
    global angle

    if contour_center is not None:
        angle = rc_utils.remap_range(contour_center[1], 0, rc.camera.get_width(), -1, 1)

    speed = 1

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
        if turnCounterMarkers < 1:
            if turnDirMarkers:
                angle = 0.6
            else:
                angle = -0.6
        else:
            turnDirMarkers = None
            turnCounterMarkers = 0
            hasBegunTurning = False

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
                    speed = 1
                    rc.drive.set_max_speed(0.8)
                    angle = rc_utils.remap_range(ar_center[1], 0, CAMERA_WIDTH, -1, 1)
                elif ar_color == "red":
                    speed = 0
                    angle = rc_utils.remap_range(ar_center[1], 0, CAMERA_WIDTH, -1, 1)

def trainEvading():
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

def update_contour():
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
        else:
            contour_center = None
            contour_area = 0

def start():
    global speed
    global angle
    global cur_state
    global color_image
    global cropped_image
    global depth_image
    global corner
    global timer
    global marker
    global flag

    cur_state = State.LineFollowing
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
    global turnDirMarkers
    global dontFollowWalls
    global trainCounter
    global overrideMaxSpeed
    global coneSpeed
    global collisionDetection

    color_image = rc.camera.get_color_image()
    cropped_image = rc_utils.crop(color_image, CROP_FLOOR[0], CROP_FLOOR[1])
    depth_image = rc.camera.get_depth_image()
    ar_markers = rc_utils.get_ar_markers(color_image)
    dt = rc.get_delta_time()

    update_contour()
    if cur_state == State.LineFollowing:
        if dt < 0.033:
            rc.drive.set_max_speed(0.325)
        elif dt < 0.085:
            rc.drive.set_max_speed(0.25)
        else:
            rc.drive.set_max_speed(0.225)
        
        if overrideMaxSpeed == None:
            if dt < 0.033:
                rc.drive.set_max_speed(0.35)
            elif dt < 0.065:
                rc.drive.set_max_speed(0.325)
            elif dt < 0.08:
                rc.drive.set_max_speed(0.3)
            else:
                rc.drive.set_max_speed(0.295)
        elif overrideMaxSpeed:
            rc.drive.set_max_speed(0.1)
        else:
            if dt < 0.033:
                rc.drive.set_max_speed(0.25)
            elif dt < 0.065:
                rc.drive.set_max_speed(0.25)
            else:
                rc.drive.set_max_speed(0.225)

        if flag == False:
            if dt < 0.033:
                rc.drive.set_max_speed(0.6)
            elif dt < 0.065:
                rc.drive.set_max_speed(0.55)
            elif dt < 0.08:
                rc.drive.set_max_speed(0.5)
            else:
                rc.drive.set_max_speed(0.4)

        if coneSpeed:
            if dt < 0.033:
                rc.drive.set_max_speed(0.275)
            elif dt < 0.085:
                rc.drive.set_max_speed(0.25)
            else:
                rc.drive.set_max_speed(0.225)

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
                if id != 4:
                    coneSpeed = False
            elif depth_image[ar_center[0]][ar_center[1]] < 300:
                id = marker.get_id()
                if id == 3:
                    cur_state = State.Elevator
        else:
            if overrideMaxSpeed != None:
                overrideMaxSpeed = False
    elif cur_state == State.WallFollowing:
        if dt < 0.033:
            rc.drive.set_max_speed(0.28)
        elif dt < 0.08:
            rc.drive.set_max_speed(0.275)
        else:
            rc.drive.set_max_speed(0.25)
        
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
    elif cur_state == State.ArMarkers:
        rc.drive.set_max_speed(0.25)
        timer += rc.get_delta_time()

        speed = 1

        for marker in ar_markers:
            if marker.get_id() == 199:
                corners = marker.get_corners()
                ar_center = ((corners[0][0] + corners[2][0]) >> 1, (corners[0][1] + corners[2][1]) >> 1)
                if depth_image[ar_center[0]][ar_center[1]] < 100:
                    if marker.get_orientation() == rc_utils.Orientation.LEFT:
                        turnDirMarkers = False
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
    elif cur_state == State.LaneFollowing:
        overrideMaxSpeed = True
        cur_state = State.LineFollowing
        lineFollowing()
    elif cur_state == State.Elevator:
        rc.drive.set_max_speed(0.25)
        elevator()
        timer += rc.get_delta_time()
        if timer > 7:
            if contour_center is not None:
                timer = 0
                cur_state = State.LineFollowing
    elif cur_state == State.ConeSlaloming:
        coneSpeed = True
        lineFollowing()
        cur_state = State.LineFollowing
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
        
        if contour_center is not None and depth_image[contour_center[0]][contour_center[1]] < 550 and timer > 5:
            cur_state = State.LineFollowing
            timer = 0
        elif ar_dist < 50:
            trainEvading()
            angle = 0
    elif cur_state == State.TileAvoiding:
        lineFollowing()
        overrideMaxSpeed = None
        cur_state = State.LineFollowing
    elif cur_state == State.LaneFollowingAccelerated:
        overrideMaxSpeed = False
        if dt < 0.033:
            rc.drive.set_max_speed(0.5)
        elif dt < 0.065:
            rc.drive.set_max_speed(0.45)
        elif dt < 0.08:
            rc.drive.set_max_speed(0.4)
        else:
            rc.drive.set_max_speed(0.35)
        lineFollowing()

    scan = rc.lidar.get_samples()
    closestPointAngle, closestPoint = rc_utils.get_lidar_closest_point(scan, (305, 55))    
    if closestPoint < 20 and cur_state != State.Elevator:
        collisionDetection = True
    
    if collisionDetection:
        if closestPoint < 40:
            speed = -1
            if closestPointAngle > 270:
                angle = -1
            else:
                angle = 1
        else:
            if closestPointAngle > 270:
                angle = 1
            else:
                angle = -1
            collisionDetection = False


    if overrideMaxSpeed == True:
        rc.drive.set_max_speed(0.1)
        speed = 0.5

    rc.drive.set_speed_angle(speed, angle)

    print(rc.get_delta_time())

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()