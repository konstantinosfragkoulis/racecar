"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Lab 3C - Depth Camera Wall Parking
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

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Add any global variables here
global kernel_size
global top_left_inclusive
global bottom_right_exclusive
global Turn_Left
global Turn_right
global can_park
global speed
global angle
global target
global left_distance
global right_distance
global major_left_angle
global major_right_angle
########################################################################################
# Functions
########################################################################################


def start():
    """
    This function is run once every time the start button is pressed
    """
    global kernel_size
    global top_left_inclusive
    global bottom_right_exclusive
    global Turn_Left
    global Turn_right
    global can_park
    global speed
    global angle
    global target
    global left_distance
    global right_distance
    global major_left_angle
    global major_right_angle

    kernel_size = 3
    top_left_inclusive= (0,0)
    bottom_right_exclusive = (rc.camera.get_height()*2//3, rc.camera.get_width())
    Turn_Left =False
    Turn_right = False
    can_park = False
    speed = 0 
    angle = 0
    target = None
    left_distance = 0
    right_distance = 0 
    major_right_angle = False
    major_left_angle = False


    # Have the car begin at a stop
    rc.drive.stop()

    # Print start message
    print(">> Lab 3C - Depth Camera Wall Parking")


def scanEdges():
    global left_distance
    global right_distance
    left_distance = rc_utils.get_pixel_average_distance(depth_image,(160,1),kernel_size)
    right_distance = rc_utils.get_pixel_average_distance(depth_image,(160,rc.camera.get_width()-1),kernel_size) 
    return left_distance , right_distance

def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    global kernel_size
    global top_left_inclusive
    global bottom_right_exclusive
    global Turn_Left
    global Turn_right
    global distance
    global can_park
    global depth_image
    global speed
    global angle
    global target
    global left_distance
    global right_distance
    global major_left_angle
    global major_right_angle

       
    depth_image = rc.camera.get_depth_image()
    depth_image = cv.GaussianBlur(depth_image,(kernel_size,kernel_size),0)
    depth_image = rc_utils.crop(depth_image,top_left_inclusive,bottom_right_exclusive)

    # TODO: Park the car 20 cm away from the closest wall with the car directly facing
    # the wall

    if not can_park:
        scanEdges()

    if left_distance == 0:
        major_right_angle = True
    if right_distance == 0:
        major_left_angle = True
    
    if left_distance < right_distance and not can_park and not major_left_angle and not major_right_angle:
        Turn_right = True
    elif right_distance < left_distance and not can_park and not major_left_angle and not major_right_angle:
        Turn_Left = True
    elif left_distance == right_distance and not can_park and not major_left_angle and not major_right_angle:
        can_park = True
    
    if major_left_angle and not can_park and not Turn_Left and not Turn_right:
        scanEdges()
        if right_distance == 0:
            speed = -0.5
            angle = 1
        else:
            can_park = True
            Turn_right = True
            major_left_angle = False

    if major_right_angle and not can_park and not Turn_Left and not Turn_right:
        scanEdges()
        if left_distance == 0:
            speed = -0.5
            angle = -1
        elif left_distance > 0:
            Turn_Left = True
            major_right_angle = False
            speed = 0
            angle = 0

    if Turn_Left and not can_park and not major_left_angle and not major_right_angle:
        left_distance = rc_utils.get_pixel_average_distance(depth_image,(160,1),kernel_size)
        right_distance = rc_utils.get_pixel_average_distance(depth_image,(160,rc.camera.get_width()-1),kernel_size)
        scanEdges()
        if left_distance - right_distance < 7:
            can_park = True
            Turn_Left = False
        else:
            speed = -0.4
            angle = -1
        
    if Turn_right and not can_park and not major_left_angle and not major_right_angle:
        left_distance = rc_utils.get_pixel_average_distance(depth_image,(160,1),kernel_size)
        right_distance = rc_utils.get_pixel_average_distance(depth_image,(160,rc.camera.get_width()-1),kernel_size)
        scanEdges()
        if right_distance - left_distance < 7:
            can_park = True
            Turn_right = False
        else:
            speed = -0.4
            angle = 1

    if can_park and not Turn_Left and not Turn_right and not major_left_angle and not major_right_angle:

        target = (160,rc.camera.get_width()//2)
        
        angle = rc_utils.remap_range(target[1],0,640,-1,1)
        distance = rc_utils.get_pixel_average_distance(depth_image,target,kernel_size)
        print(distance)           
        if distance > 80: 
            speed = 0.35
        elif distance < 80 and distance> 35:
            speed = 0.2
        elif distance < 35 and distance > 21:
            speed = 0.1
        elif distance < 21  and distance > 19:
            speed = 0
        elif distance < 19.5:
            speed = -0.3 
    rc.drive.set_speed_angle(speed,angle)

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()

