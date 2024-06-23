"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Lab 5 - AR Markers
"""

########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv
import numpy as np

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils
from enum import Enum

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()
full_course = False
angle = 0.0
speed = 1.0
cnt = 0.0
mode = "normal"
RED = ((0, 50, 50), (10, 255, 255))      # The HSV range for the color red
GREEN = ((40, 50, 50), (80, 255, 255))  # The HSV range for the color green
BLUE = ((90, 50, 50), (120, 255, 255))   # The HSV range for the color blue
CROP_FLOOR = ((360, 0), (rc.camera.get_height(), rc.camera.get_width()))
LEFT_WINDOW = (260, 280)
RIGHT_WINDOW = (80, 100)
LEFT45_WINDOW = (305, 325)
RIGHT45_WINDOW = (35, 55)
GREEN = ((36, 50, 50), (80, 255, 255))
TURN_ANGLE = 4.9
times = 0
# Add any global variables here

########################################################################################
# Functions
########################################################################################

def start():
    """
    This function is run once every time the start button is pressed
    """
    # Have the car begin at a stop
    rc.drive.stop()

    # Print start message
    print(">> Lab 5 - AR Markers")


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    global speed
    global angle
    global mode
    global cnt
    global BLUE
    global RED
    global GREEN
    global CROP_FLOOR
    global full_course
    global times

    color_image = rc.camera.get_color_image()
    cropped_image = rc_utils.crop(color_image, CROP_FLOOR[0], CROP_FLOOR[1])
    depth_image = rc.camera.get_depth_image()
    markers = rc_utils.get_ar_markers(color_image)
    scan = rc.lidar.get_samples()
    _, leftDist = rc_utils.get_lidar_closest_point(scan, LEFT_WINDOW)
    _, rightDist = rc_utils.get_lidar_closest_point(scan, RIGHT_WINDOW)
    _, left45Dist = rc_utils.get_lidar_closest_point(scan, LEFT45_WINDOW)
    _, right45Dist = rc_utils.get_lidar_closest_point(scan, RIGHT45_WINDOW)

    if left45Dist > right45Dist:
        angle = rc_utils.clamp(right45Dist - left45Dist + 5, -1, 0)
    elif right45Dist > left45Dist:
        angle = rc_utils.clamp(right45Dist - left45Dist - 5, 0, 1)
    elif leftDist > rightDist:
        angle = rc_utils.clamp(rightDist - leftDist + 1, -1, 0)
    elif rightDist > leftDist:
        angle = rc_utils.clamp(rightDist - leftDist - 1, 0, 1)
    # TODO: Turn left if we see a marker with ID 0 and right for ID 1

    if len(markers) != 0 and mode == "normal":
        if depth_image[markers[0].get_corners()[0][0]][markers[0].get_corners()[0][1]] < 60:
            if markers[0].get_id() == 0:
                mode = "left"
            elif markers[0].get_id() == 1:
                mode = "right"
            elif markers[0].get_id() == 199:
                if markers[0].get_orientation().value == 1:
                    mode = "left"
                else:
                    mode = "right"
            elif markers[0].get_id() == 2:
                markers[0].detect_colors(color_image, [(RED[0], RED[1], "red"), (GREEN[0], GREEN[1], "green"), (BLUE[0], BLUE[1], "blue")])
                if markers[0].get_color() == "blue":
                    mode = "follow_blue"
                elif markers[0].get_color() == "red":
                    mode = "follow_red"
                else:
                    mode = "follow_green"
        elif not full_course:
            angle = 0

    if times == 2:
        speed = 0
            
    
    if mode == "left":
        cnt += rc.get_delta_time()
        if cnt < .5:
            angle = -1
        elif cnt > 3 and not full_course:
            angle = 0
            mode = "normal"
            cnt = 0
        elif cnt > TURN_ANGLE and times == 0:
            angle = -1
            if cnt > 5.3:
                angle = 0
                mode = "normal"
                cnt = 0
                times += 1
        elif cnt > TURN_ANGLE + 1 and times == 1:
            angle = -1
            if cnt > 6.3:
                angle = 0
                mode = "normal"
                cnt = 0
    
    if mode == "right":
        cnt += rc.get_delta_time()
        if cnt < .5:
            angle = 1
        elif cnt > 3 and not full_course:
            angle = 0
            mode = "normal"
            cnt = 0
        elif cnt > TURN_ANGLE and times == 0:
            angle = 1
            if cnt > 5.3:
                angle = 0
                mode = "normal"
                cnt = 0
                times += 1
        elif cnt > TURN_ANGLE + 1 and times == 1:
            angle = 1
            if cnt > 6.3:
                angle = 0
                mode = "normal"
                cnt = 0

    if mode == "follow_blue":
        cnt += rc.get_delta_time()
        contours = rc_utils.find_contours(cropped_image, BLUE[0], BLUE[1])
        contour = rc_utils.get_largest_contour(contours, 30)
        contour_center = rc_utils.get_contour_center(contour)
        angle = 1 if cnt < .5 else rc_utils.remap_range(contour_center[1], 0, 768, -1, 1)
        if cnt > 3 and not full_course:
            angle = 0
            mode = "normal"
            cnt = 0
            full_course = True
        elif cnt > TURN_ANGLE:
            angle = 1
            if cnt > 5.5:
                angle = 0
                mode = "normal"
                cnt = 0
                times += 1
        

    if mode == "follow_red":
        cnt += rc.get_delta_time()
        contours = rc_utils.find_contours(cropped_image, RED[0], RED[1])
        contour = rc_utils.get_largest_contour(contours, 30)
        contour_center = rc_utils.get_contour_center(contour)
        angle = -1 if cnt < .5 else rc_utils.remap_range(contour_center[1], 0, 768, -1, 1)
        if cnt > 3 and not full_course:
            angle = 0
            mode = "normal"
            cnt = 0
        elif cnt > TURN_ANGLE:
            angle = -1
            if cnt > 5.5:
                angle = 0
                mode = "normal"
                cnt = 0
                times += 1

        

    if mode == "follow_green":
        cnt += rc.get_delta_time()
        contours = rc_utils.find_contours(color_image, RED[0], RED[1])
        contour = rc_utils.get_largest_contour(contours, 30)
        contour_center = rc_utils.get_contour_center(contour)
        rc.display.show_color_image(color_image)
        if contour_center is not None:
            angle = rc_utils.remap_range(contour_center[1], 0, 768, -1, 1)
            if angle > 0.001 and angle + 0.1 <= 1:
                angle += 0.1
            elif angle < -0.001 and angle - 0.2 >= -1:
                angle -= 0.2
        if cnt > 3 and not full_course:
            angle = 0
            mode = "normal"
            cnt = 0
        

    
    # TODO: If we see a marker with ID 199, turn left if the marker faces left and right
    # if the marker faces right

    # TODO: If we see a marker with ID 2, follow the color line which matches the color
    # border surrounding the marker (either blue or red). If neither color is found but
    # we see a green line, follow that instead.

    print(f"Mode: {mode}, cnt: {cnt}, full_course: {full_course}")
    rc.display.show_color_image(color_image)
    rc.drive.set_speed_angle(speed, angle)


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()

