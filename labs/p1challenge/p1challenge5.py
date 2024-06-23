"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Phase 1 Challenge - Cone Slaloming
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
from enum import IntEnum

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Add any global variables here
global curr_state
global color
global ready_to_increment
global timer
global num_passed_cones
# >> Constants
# The smallest contour we will recognize as a valid contour
MIN_CONTOUR_AREA = 750

# The HSV range for the color orange, stored as (hsv_min, hsv_max)
RED = ((170, 50, 50), (10, 255, 255))
BLUE = ((100,150,150), (120, 255, 255))

# >> Variables
speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels
contour_center = None  # The (pixel row, pixel column) of contour
contour_area = 0  # The area of contour
color = 'none'
ready_to_increment = False
timer = 0
num_passed_cones = 0
########################################################################################
# Functions
########################################################################################
class State(IntEnum):
    init_approach = 0
    cone_align = 1
    pass_red = 2
    pass_blue = 3
    cone_12 = 4

curr_state: State = State.init_approach

def update_contour():
    """
    Finds contours in the current color image and uses them to update contour_center
    and contour_area
    """
    global contour_center
    global contour_area
    global color
    
    image = rc.camera.get_color_image() 

    if image is None:
        contour_center = None
        contour_area = 0
    else:
        # Find all of the red/blue contours
        contours_red = rc_utils.find_contours(image, RED[0], RED[1])
        contours_blue = rc_utils.find_contours(image, BLUE[0], BLUE[1])
        
        # Select the largest contour
        contour_red = rc_utils.get_largest_contour(contours_red, MIN_CONTOUR_AREA)
        contour_blue = rc_utils.get_largest_contour(contours_blue, MIN_CONTOUR_AREA)
        
        if contour_red is not None and contour_blue is not None:
            if rc_utils.get_contour_area(contour_red) > rc_utils.get_contour_area(contour_blue):
                contour = contour_red
                color = 'red'                
            else:
                contour = contour_blue 
                color = 'blue' 
        elif contour_red is not None:
            contour = contour_red
            color = 'red'
            
        elif contour_blue is not None:
            contour = contour_blue
            color = 'blue'

        else:
            contour = None
            color = 'none'

        if contour is not None:
            # Calculate contour information
            contour_center = rc_utils.get_contour_center(contour)
            contour_area = rc_utils.get_contour_area(contour)

            # Draw contour onto the image
            rc_utils.draw_contour(image, contour)
            rc_utils.draw_circle(image, contour_center)
        else:
            contour_center = None
            contour_area = 0

        # Display the image to the screen
        rc.display.show_color_image(image)
        

def start():
    """
    This function is run once every time the start button is pressed
    """
    global curr_state
    # Have the car begin at a stop
    rc.drive.stop()
    curr_state = State.init_approach
    # Print start message
    print(">> Phase 1 Challenge: Cone Slaloming")


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    global curr_state
    global color
    global ready_to_increment
    global timer
    global num_passed_cones

    # TODO: Slalom between red and blue cones.  The car should pass to the right of
    # each red cone and the left of each blue cone.

    update_contour()
    depth_image = rc.camera.get_depth_image()
    depth_image = (depth_image - 0.01) % 10000

    color_image = rc.camera.get_color_image()
    if curr_state == State.init_approach:
        #ready_to_wait = True
        speed = 1
        angle = 0.07
        #print("Distance: ", depth_image[contour_center[0], contour_center[1]])
        # if depth_image[contour_center[0], contour_center[1]] < 100:
        #     num_passed_cones += 1
        #     curr_state = State.cone_align
        if depth_image[contour_center[0], contour_center[1]] < 60:
            num_passed_cones += 1
            if color == 'blue':
                curr_state = State.pass_blue
                #print("State: ", curr_state)
            elif color == 'red':
                curr_state = State.pass_red
                #print("State: ", curr_state)
    
    if curr_state == State.pass_red:
        if num_passed_cones >= 11:
            curr_state = State.cone_12

        elif color == 'blue' and depth_image[contour_center[0], contour_center[1]] < 120:
            #timer = 0
            ready_to_increment = True
            curr_state = State.cone_align 
            
        else:    
            speed = 1
            if contour_center is not None and color == 'red':
                angle = rc_utils.remap_range(contour_center[1], 0, rc.camera.get_width() // 2, 0.8, 1) #-200, 0.7, 1
                angle = rc_utils.clamp(angle, 0.8, 1) #0.7
            
                # timer += rc.get_delta_time()
                # angle = 0.1
                # if timer >= 0.25:
            
            elif color == 'blue':
                if ready_to_increment:
                    num_passed_cones += 1
                    ready_to_increment = False
                print('angle: -0.1')
                angle = -0.2
            else:
                if ready_to_increment:
                    num_passed_cones += 1
                    ready_to_increment = False
                print("angle: -0.5")
                angle = -0.45

        
    if curr_state == State.pass_blue:
        if num_passed_cones >= 11:
            curr_state = State.cone_12

        elif color == 'red' and depth_image[contour_center[0], contour_center[1]] < 120:
            #timer = 0
            ready_to_increment = True
            curr_state = State.cone_align
             
        else:    
            speed = 1
            if contour_center is not None and color == 'blue':
                angle = rc_utils.remap_range(contour_center[1], rc.camera.get_width() // 2, rc.camera.get_width(), -1, -0.8) #200, ,-1, -0.7
                angle = rc_utils.clamp(angle, -1, -0.8) #-0.7
            # else:  
            #     timer += rc.get_delta_time()
            #     angle = -0.1
            #     if timer >= 0.25:
            elif color == 'red':
                if ready_to_increment:
                    num_passed_cones += 1
                    ready_to_increment = False
                print('angle: 0.1')
                angle = 0.2
            else:
                if ready_to_increment:
                    num_passed_cones += 1
                    ready_to_increment = False
                print("angle: 0.5")
                angle = 0.45
   
    if curr_state == State.cone_align:
        """
        if num_passed_cones > 10:
            curr_state = State.cone_12
        """
        #else:
        speed = 1
        if color == 'red':
            angle = rc_utils.remap_range(contour_center[1] + 45, 100, rc.camera.get_width(), -1, 1) #-120 added 40
            angle = rc_utils.clamp(angle, -1, 1)
        if color == 'blue':
            angle = rc_utils.remap_range(contour_center[1] - 45, 0, rc.camera.get_width() -100, -1, 1) #+120 minus 40
            angle = rc_utils.clamp(angle, -1, 1)
        if depth_image[contour_center[0], contour_center[1]] < 90 and color == 'blue': #90
            #num_passed_cones += 1
            curr_state = State.pass_blue
            
        elif depth_image[contour_center[0], contour_center[1]] < 90 and color == 'red': #90
            #num_passed_cones += 1
            curr_state = State.pass_red
    
    if curr_state == State.cone_12:
        timer += rc.get_delta_time()
        speed = 1
        angle = 0.45
        if timer >= 0.2:
            angle = -1
        if contour_center is not None and depth_image[contour_center[0], contour_center[1]] < 120: #100
            curr_state = State.cone_align
            num_passed_cones = 0
           
    print("Current state: ,", curr_state)      
    print(f"Passed {num_passed_cones} cones") 
    
    rc.drive.set_speed_angle(speed, angle)
########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()




"""
get both images
go into start state
    find the center of the nearest cone
    drive towards it head on until depth camera says center of nearest cone is X cm
    activate turn outwards state based on color
go into turn outwards state
    if cone was red
        turn right Y degrees 
        until contour center is none (out of screen)
        activate red turn inwards state
    if cone was blue
        turn left Y degrees
        until contour center is none (out of screen)
        activate blue turn inwards state
go into red inwards state
    turn left until the center of next cone is found
    keep turning until center of next cone is on the right side of the screen
    then go straight 
go into approach state
    keep turning until the cone moves to the ride side of vision



go into blue pass state
    turn right Y degrees
    




"""
