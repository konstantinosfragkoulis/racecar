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
BLUE = ((100, 100, 100), (110, 255, 255))

potentialColors = [
    ((130, 150, 150), (140, 255, 255), "purple"),
    ((10, 50, 50), (20, 255, 255), "orange"),
    ((170, 50, 50), (10, 255, 255), "red"),
    ((40, 50, 50), (80, 255, 255), "green"),
    ((100, 150, 50), (110, 255, 255), "blue")
]

CROP_FLOOR = ((200, 0), (rc.camera.get_height(), rc.camera.get_width()))
LEFT45_WINDOW = (305, 325)
RIGHT45_WINDOW = (35, 55)
TURN_ANGLE = 4.9

CAMERA_HEIGHT = rc.camera.get_height()
CAMERA_WIDTH = rc.camera.get_width()
CAMERA_HEIGHTd2 = CAMERA_HEIGHT >> 1
CAMERA_WIDTHd2 = CAMERA_WIDTH >> 1

CROP_FLOOR_LINE = ((300, 0), (CAMERA_HEIGHT, CAMERA_WIDTH))
CROP_FLOOR_LANE_ACCEL = ((200, 0), (CAMERA_HEIGHT, CAMERA_WIDTH))
CROP_RIGHT = ((300, CAMERA_WIDTHd2), (CAMERA_HEIGHT, CAMERA_WIDTH))
CROP_LEFT = ((300, 0), (CAMERA_HEIGHT, CAMERA_WIDTHd2))
CROP_RIGHT_ACCELERATED = ((0, CAMERA_WIDTHd2), (CAMERA_HEIGHT, CAMERA_WIDTH))
CROP_LEFT_ACCELERATED = ((0, 0), (CAMERA_HEIGHT, CAMERA_WIDTHd2))

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

class LaneState(enum.IntEnum):
    FollowNormalLane = 0
    TurnExtreme = 1
    GoToSecondary = 2

cur_state = State.LineFollowing
cur_mode = Mode.Normal
cur_lane_state = LaneState.FollowNormalLane

laneColor = None
curLaneColor = None

change_lane_counter = 0

hax1 = 0
_angle = 0

firstTurn = False
override = False
turn_right = True

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

    return (contourP, contourO)

def laneFollowing():
    global speed, angle, curLaneColor, laneColor, change_lane_counter, cur_lane_state, turn_right, cur_state
    speed = .8

    color_image_lane = rc.camera.get_color_image()
    depth_image_lane = rc.camera.get_depth_image()
    contours_purple = rc_utils.find_contours(color_image_lane, PURPLE[0], PURPLE[1])
    contours_purple_list = []
    contours_orange = rc_utils.find_contours(color_image_lane, ORANGE[0], ORANGE[1])
    contours_orange_list = []

    for contour in contours_purple:
        if rc_utils.get_contour_area(contour) > 400:
            contours_purple_list.append(contour)
    
    for contour in contours_orange:
        if rc_utils.get_contour_area(contour) > 400:
            contours_orange_list.append(contour)
    
    contours_purple = tuple(contours_purple_list)
    contours_orange = tuple(contours_orange_list)

    def get_largest_contour_lane():
        global laneColor
        nonlocal contours_purple, contours_orange
        largest_contour = None

        if laneColor == "purple":
            largest_contour = rc_utils.get_largest_contour(contours_purple)
            if largest_contour is not None:
                contours_purple = tuple([x for x in contours_purple if (x.all() != largest_contour.all())])
                return rc_utils.get_contour_center(largest_contour)
            else:
                return None
        else:
            largest_contour = rc_utils.get_largest_contour(contours_orange)
            if largest_contour is not None:
                contours_orange = tuple([x for x in contours_orange if (x.all() != largest_contour.all())])
                return rc_utils.get_contour_center(largest_contour)
            else:
                return None
            
    def get_contour_centers(contours):
        return [rc_utils.get_contour_center(contour) for contour in contours]

    if cur_lane_state == LaneState.FollowNormalLane:
        largest_contour_center = get_largest_contour_lane()
        second_largest_contour_center = get_largest_contour_lane()
        if largest_contour_center is not None:
            if second_largest_contour_center is not None:
                mid_contour_center = ((largest_contour_center[0] + second_largest_contour_center[0])>>1, (largest_contour_center[1] + second_largest_contour_center[1])>>1)
                angle = rc_utils.remap_range(mid_contour_center[1], 0, CAMERA_WIDTH, -1, 1)
            else:
                angle = rc_utils.remap_range(largest_contour_center[1], 0, CAMERA_WIDTH, -1, 1)
        else:
            if second_largest_contour_center is not None:
                angle = rc_utils.remap_range(second_largest_contour_center[1], 0, CAMERA_WIDTH, -1, 1)
        
        if laneColor == "purple" and contours_orange is not None:
            for contour in contours_orange:
                if contour is not None:
                    c_center = rc_utils.get_contour_center(contour)
                    if c_center is not None and depth_image_lane[c_center[0]][c_center[1]] < 50:
                        cur_lane_state = LaneState.TurnExtreme
        if laneColor == "orange" and contours_purple is not None:
            for contour in contours_purple:
                if contour is not None:
                    c_center = rc_utils.get_contour_center(contour)
                    if c_center is not None and depth_image_lane[c_center[0]][c_center[1]] < 50:
                        cur_lane_state = LaneState.TurnExtreme
        angle *= 1.2
    
    elif cur_lane_state == LaneState.TurnExtreme:
        if turn_right:
            angle = 1
        else:
            angle = -1

        there_are_close = False
        if laneColor == "purple" and contours_orange is not None:
            for contour in contours_orange:
                if contour is not None:
                    c_center = rc_utils.get_contour_center(contour)
                    if c_center is not None and depth_image_lane[c_center[0]][c_center[1]] < 50:
                        there_are_close = True
        if laneColor == "orange" and contours_purple is not None:
            for contour in contours_purple:
                if contour is not None:
                    c_center = rc_utils.get_contour_center(contour)
                    if c_center is not None and depth_image_lane[c_center[0]][c_center[1]] < 50:
                        there_are_close = True
        
        if turn_right and not there_are_close:
            cur_lane_state = LaneState.GoToSecondary
        if not turn_right:
            if (laneColor == "purple" and contours_orange is None or not len(contours_orange) or min([depth_image[c_center[0]][c_center[1]] for c_center in get_contour_centers(contours_orange)]) > 500) or \
                (laneColor == "orange" and contours_purple is None or not len(contours_purple) or min([depth_image[c_center[0]][c_center[1]] for c_center in get_contour_centers(contours_purple)]) > 500):
                contour_center_green = rc_utils.get_contour_center(rc_utils.get_largest_contour(rc_utils.find_contours(color_image_lane, GREEN[0], GREEN[1])))
                if contour_center_green is not None and contour_center_green != (None, None) and depth_image_lane[contour_center_green[0]][contour_center_green[1]] < 200:
                    angle = rc_utils.remap_range(contour_center_green[1], 0, CAMERA_WIDTH, -1, 1)
                    #if depth_image_lane[contour_center_green[0]][contour_center_green[1]] < 50:
                    #    cur_state = State.LineFollowing
                    cur_state = State.LineFollowing
    
    elif cur_lane_state == LaneState.GoToSecondary:
        turn_right = False
        if curLaneColor == "purple":
            largest_secondary_contour_center = rc_utils.get_contour_center(rc_utils.get_largest_contour(contours_orange))
        else:
            largest_secondary_contour_center = rc_utils.get_contour_center(rc_utils.get_largest_contour(contours_purple))
        if largest_secondary_contour_center is None or largest_secondary_contour_center == (None, None):
            #cur_lane_state = LaneState.TurnExtreme
            #turn_right = False
            pass
        else:
            angle = rc_utils.remap_range(largest_secondary_contour_center[1] + 200, 0, CAMERA_WIDTH, -1, 1, True)
            #if depth_image_lane[largest_secondary_contour_center[0]][largest_secondary_contour_center[1]] < 32 or (len(contours_orange) < 2 if laneColor == "purple" else len(contours_purple) < 2):
            if laneColor == "purple":
                if len(contours_orange) == 0:
                    angle = .5
                elif len(contours_orange) == 1:
                    c_center = rc_utils.get_contour_center(contours_orange[0])
                    if depth_image[c_center[0]][c_center[1]] < 50:
                        cur_lane_state = LaneState.TurnExtreme

            else:
                if len(contours_purple) == 0:
                    angle = .25
                elif len(contours_purple) == 1:
                    c_center = rc_utils.get_contour_center(contours_purple[0])
                    if depth_image[c_center[0]][c_center[1]] < 50:
                        cur_lane_state = LaneState.TurnExtreme
                                    
    
    print(cur_lane_state, angle)
    #print(len(contours_purple) if laneColor == "orange" else len(contours_orange))
    contours_green = rc_utils.find_contours(color_image_lane, GREEN[0], GREEN[1])
    for contour in contours_green:
        rc_utils.draw_contour(color_image_lane, contour)

    angle = rc_utils.clamp(angle, -1, 1)
    rc.display.show_color_image(color_image_lane)
    if len(contours_orange):
        print(f"min orange: {min([depth_image[c_center[0]][c_center[1]] for c_center in get_contour_centers(contours_orange)])}")
    if len(contours_purple):
        print(f"min purple: {min([depth_image[c_center[0]][c_center[1]] for c_center in get_contour_centers(contours_purple)])}")

def arMarkers():
    pass

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

def coneSlaloming():
    global speed
    global angle

def trainEvading():
    global speed
    global angle
    
    print("TRAIN EVADING")
    speed = angle = 0

def tileAvoiding():
    global speed
    global angle
    
    print("TILE AVOIDING")
    speed = angle = 0

def laneFollowingAccelerated():
    global speed, angle, timer

    color_image_new = rc.camera.get_color_image()
    cropped_image_new = rc_utils.crop(color_image_new, CROP_FLOOR_LANE_ACCEL[0], CROP_FLOOR_LANE_ACCEL[1])
    speed = 1
    contours = rc_utils.find_contours(cropped_image_new, BLUE[0], BLUE[1])
    contour_blue = rc_utils.get_largest_contour(contours, 600)
    if contour_blue is not None and not (None in contour_blue):
        contour_center_blue = rc_utils.get_contour_center(contour_blue)
        angle1 = rc_utils.remap_range(contour_center_blue[1], 0, CAMERA_WIDTH, -1, 1)
        contours = tuple([x for x in contours if x.all() != contour_blue.all()]) #contours.remove(contour_blue)
        rc_utils.draw_circle(cropped_image_new, contour_center_blue)

        contour_blue_second = rc_utils.get_largest_contour(contours, 600)
        if contour_blue_second is not None and not (None in contour_blue_second):
            contour_center_blue_second = rc_utils.get_contour_center(contour_blue_second)
            angle2 = rc_utils.remap_range(contour_center_blue_second[1], 0, CAMERA_WIDTH, -1, 1)
            rc_utils.draw_circle(cropped_image_new, contour_center_blue_second)
            #mid_contour_center = ((contour_center_blue[0] + contour_center_blue_second[0])>>1, (contour_center_blue[1] + contour_center_blue_second[1])>>1)
            #rc_utils.draw_circle(cropped_image_new, mid_contour_center)
            #angle = rc_utils.remap_range(mid_contour_center[1], 0, CAMERA_WIDTH, -1, 1)
            if timer < 12:
                angle = angle1*.4 + angle2*.6
            else:
                angle = angle1*.6 + angle2*.4
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

    color_image = rc.camera.get_color_image()
    cropped_image = rc_utils.crop(color_image, CROP_FLOOR[0], CROP_FLOOR[1])
    depth_image = rc.camera.get_depth_image()
    ar_markers = rc_utils.get_ar_markers(color_image)

    for i in range(1, 500):
        if i == 499:
            print("i:", i)
        for j in range(1, 200):
            if j == 199 and i == 499:
                print("j:", j)
            for k in range(1, 200):
                if k == 199 and j == 99 and i == 499:
                    print("k:", k)
                var = (i/j/k)

    update_contour()
    if cur_state == State.LineFollowing:
        rc.drive.set_max_speed(0.25)
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
                        #cur_state = State.ArMarkers
                        pass
                    else:
                        cur_state = State.WallFollowing
                        flag = True
                
                elif id == 1:
                    cur_state = State.LaneFollowing
                elif id == 3:
                    cur_state = State.Elevator
                elif id == 4:
                    cur_state = State.ConeSlaloming
                elif id == 5:
                    cur_state = State.TrainEvading
                elif id == 6:
                    cur_state = State.TileAvoiding
                elif id == 8:
                    cur_state = State.LaneFollowingAccelerated
            elif depth_image[ar_center[0]][ar_center[1]] < 300:
                id = marker.get_id()
                if id == 3:
                    cur_state = State.Elevator
    # Done
    elif cur_state == State.WallFollowing:
        rc.drive.set_max_speed(0.28)
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
        #rc.drive.set_max_speed(0.25)
        #timer += rc.get_delta_time()
        #if timer > 1:
        #    if contour_center is None:
        #        arMarkers()
        #    else:
        #        timer = 0
        #        cur_state = State.LineFollowing
        #else:
        #    speed = 1
        #    angle = 0
        lineFollowing()
    # Done
    elif cur_state == State.LaneFollowing:
        rc.drive.set_max_speed(.25)
        if ar_markers:
            ar_markers[0].detect_colors(color_image, potentialColors)
            laneColor = ar_markers[0].get_color()
        timer += rc.get_delta_time()
        if timer > 1:
            laneFollowing()
            
        else:
            speed = 1
            angle = 0
    elif cur_state == State.Elevator:
        rc.drive.set_max_speed(0.25)
        elevator()
        timer += rc.get_delta_time()
        if timer > 7:
            if contour_center is not None:
                timer = 0
                cur_state = State.LineFollowing
    elif cur_state == State.ConeSlaloming:
        #rc.drive.set_max_speed(0.25)
        #timer += rc.get_delta_time()
        #if timer > 2:
        #    if contour_center is None:
        #        coneSlaloming()
        #    else:
        #        timer = 0
        #        cur_state = State.LineFollowing
        #else:
        #    speed = 1
        #    angle = 0.25
        lineFollowing()
        rc.drive.set_max_speed(0.15)
        cur_state = State.LineFollowing
    elif cur_state == State.TrainEvading:
        rc.drive.set_max_speed(0.25)
        #trainEvading()
        lineFollowing()
    elif cur_state == State.TileAvoiding:
        rc.drive.set_max_speed(0.25)
        #tileAvoiding()
        lineFollowing()
    # TODO: incomplete
    #KALOUMENE HELPPPP
    elif cur_state == State.LaneFollowingAccelerated:
        timer += rc.get_delta_time()
        
        if timer > 16:
            rc.drive.set_max_speed(.8)
            if contour_center is not None and depth_image[contour_center[0]][contour_center[1]] < 100:
                cur_state = State.LineFollowing
        
        laneFollowingAccelerated()
        
    rc.drive.set_speed_angle(speed, angle)

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
