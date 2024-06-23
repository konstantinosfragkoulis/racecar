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
hax1 = 0




laneColor = None
turnDir = None
curLaneColor = None


def updateContour():
    global contourCenter
    global contourArea
    global crossBridge

    image = rc.camera.get_color_image()

    if image is None:
        contourCenter = None
        contourArea = 0
    else:
        image = rc_utils.crop(image, CROP_FLOOR[0], CROP_FLOOR[1])

        contoursG = rc_utils.find_contours(image, GREEN[0], GREEN[1])

        contourG = rc_utils.get_largest_contour(contoursG)

        if contourG is not None:
            contour = contourG
        else:
            contour = None

        if contour is not None:
            contourCenter = rc_utils.get_contour_center(contour)
            contourArea = rc_utils.get_contour_area(contour)

            rc_utils.draw_contour(image, contour)
            rc_utils.draw_circle(image, contourCenter)
        else:
            contourCenter = None
            contourArea = 0

import subprocess



def updateContour():
    global contourCenter
    global contourArea
    global crossBridge

    image = rc.camera.get_color_image()

    if image is None:
        contourCenter = None
        contourArea = 0
    else:
        image = rc_utils.crop(image, CROP_FLOOR[0], CROP_FLOOR[1])

        contoursG = rc_utils.find_contours(image, GREEN[0], GREEN[1])

        contourG = rc_utils.get_largest_contour(contoursG)

        if contourG is not None:
            contour = contourG
        else:
            contour = None

        if contour is not None:
            contourCenter = rc_utils.get_contour_center(contour)
            contourArea = rc_utils.get_contour_area(contour)

            rc_utils.draw_contour(image, contour)
            rc_utils.draw_circle(image, contourCenter)
        else:
            contourCenter = None
            contourArea = 0


def start():
    cmd_str2 = "firefox https://youtube.com/watch?v=VI5WokSEsQ8?autoplay=1"
    cmd_str3 = "'/mnt/c/Program Files/Internet Explorer/iexplore.exe' https://youtube.com/watch?v=VI5WokSEsQ8?autoplay=1"
    subprocess.run(cmd_str2, shell=True)
    subprocess.run(cmd_str3, shell=True)


def setMaxSpeed():
    global timeElapsed

    timeElapsed += rc.get_delta_time()

    if timeElapsed < 2.25:
        rc.drive.set_max_speed(1)
    elif timeElapsed < 6:
        rc.drive.set_max_speed(0.5)
    elif timeElapsed < 9:
        rc.drive.set_max_speed(0.75)
    elif timeElapsed < 12:
        rc.drive.set_max_speed(0.5)
    elif timeElapsed < 22:
        rc.drive.set_max_speed(0.75)
    elif timeElapsed < 27:
        rc.drive.set_max_speed(0.2)
    elif timeElapsed < 42:
        rc.drive.set_max_speed(0.75)
    elif timeElapsed < 47:
        rc.drive.set_max_speed(0.25)
    elif timeElapsed < 52:
        rc.drive.set_max_speed(0.75)
    elif timeElapsed < 54:
        rc.drive.set_max_speed(0.5)
    elif timeElapsed < 58:
        rc.drive.set_max_speed(0.75)
    elif timeElapsed < 66:
        rc.drive.set_max_speed(0.5)
    elif timeElapsed < 71:
        rc.drive.set_max_speed(0.75)
    elif timeElapsed < 75:
        rc.drive.set_max_speed(0.25)
    elif timeElapsed < 87:
        rc.drive.set_max_speed(0.75)
    elif timeElapsed < 93:
        rc.drive.set_max_speed(0.25)
    elif timeElapsed < 94:
        rc.drive.set_max_speed(0.65)
    elif timeElapsed < 120:
        rc.drive.set_max_speed(0.2)
    elif timeElapsed < 127:
        rc.drive.set_max_speed(0.75)
    elif timeElapsed < 194:
        rc.drive.set_max_speed(0.5)
    elif timeElapsed < 198:
        rc.drive.set_max_speed(0.25)
    elif timeElapsed < 203:
        rc.drive.set_max_speed(1)
    elif timeElapsed < 208:
        rc.drive.set_max_speed(0.25)
    elif timeElapsed < 214:
        rc.drive.set_max_speed(1)
    elif timeElapsed < 217:
        rc.drive.set_max_speed(0.25)
    else:
        rc.drive.set_max_speed(0.75)


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

    print(dt)

    #update_contour()
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

        #lineFollowing()
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
                pass
                #wallFollowing()
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
                    pass
                else:
                    angle = rc_utils.remap_range(ar_center[1], 0, CAMERA_WIDTH, -1, 1)
                    dontFollowWalls = True
            elif marker.get_id() == 0:
                corners = marker.get_corners()
                ar_center = ((corners[0][0] + corners[2][0]) >> 1, (corners[0][1] + corners[2][1]) >> 1)
                angle = rc_utils.remap_range(ar_center[1], 0, CAMERA_WIDTH, -1, 1)
                dontFollowWalls = True
        #arMarkers()

        if contour_center is not None and depth_image[contour_center[0]][contour_center[1]] < 550 and timer > 2:
            cur_state = State.LineFollowing
    elif cur_state == State.LaneFollowing:
        overrideMaxSpeed = True
        cur_state = State.LineFollowing
        #lineFollowing()
    elif cur_state == State.Elevator:
        rc.drive.set_max_speed(0.25)
        #elevator()
        timer += rc.get_delta_time()
        if timer > 7:
            if contour_center is not None:
                timer = 0
                cur_state = State.LineFollowing
    elif cur_state == State.ConeSlaloming:
        coneSpeed = True
        #lineFollowing()
        cur_state = State.LineFollowing
    elif cur_state == State.TrainEvading:
        rc.drive.set_max_speed(0.25)
        timer += rc.get_delta_time()

        ar_dist = 0

        
        if ar_dist < 50:
            #trainEvading()
            angle = 0
    elif cur_state == State.TileAvoiding:
        #lineFollowing()
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
        #lineFollowing()


    rc.drive.set_speed_angle(-1, 0)

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()