"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020
 
Lab 3A - Depth Camera Safety Stop
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
 
isAbleToDrive = True
override = False
speed = 0
startStopping = False
distanceFromObj = 0
counter = 0
stop = False
shouldDeductSpeed = True
setSpeedTo0 = False
counter = 0.0
hax = False
 
########################################################################################
# Functions
########################################################################################
 
 
def start():
    """
    This function is run once every time the start button is pressed
    """
 
    global isAbleToDrive
    global override
    global speed
    global startStopping
    global distanceFromObj
    global counter
    global stop
    global shouldDeductSpeed
    global setSpeedTo0
    global centerOfScreen
 
    isAbleToDrive = True
    override = False
    speed = 0
    startStopping = False
    distanceFromObj = 0
    counter = 0
    stop = False
    shouldDeductSpeed = True
    setSpeedTo0 = False
    centerOfScreen = (rc.camera.get_height() // 2, rc.camera.get_width() // 2);
 
    # Have the car begin at a stop
    rc.drive.stop()
 
 
    # Print start message
    print(
        ">> Lab 3A - Depth Camera Safety Stop\n"
        "\n"
        "Controls:\n"
        "    Right trigger = accelerate forward\n"
        "    Right bumper = override safety stop\n"
        "    Left trigger = accelerate backward\n"
        "    Left joystick = turn front wheels\n"
        "    A button = print current speed and angle\n"
        "    B button = print the distance at the center of the depth image"
    )
 
 
def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
 
    global isAbleToDrive
    global override
    global speed
    global startStopping
    global distanceFromObj
    global counter
    global stop
    global shouldDeductSpeed
    global setSpeedTo0
    global centerOfScreen
    global counter
    global hax
 
 
    ### GET INPUT FROM TRIGGERS ###
 
    # Use the triggers to control the car's speed
    rt = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
    lt = rc.controller.get_trigger(rc.controller.Trigger.LEFT)
    if isAbleToDrive or override:
        speed = rt - lt
 
    ### GET INPUT FROM TRIGGERS ###
 
 
 
 
    ### FIND THE DISTANCE OF THE CENTER OBJECT ###
 
    # Calculate the distance of the object directly in front of the car
    depth_image = rc.camera.get_depth_image()
    center_distance = rc_utils.get_depth_image_center_distance(depth_image)
 
    ### FIND THE DISTANCE OF THE CENTER OBJECT ###
 
 
 
 
    # Allow the user to override safety stop by holding the right bumper.
    if rc.controller.is_down(rc.controller.Button.RB):
        override = True
    else:
        override = False
 
 
 
 
    ### DEPTH CAMERA - FIND CLOSEST PIXEL ###
 
    # Capture a depth image to find the closest pixel
    depth_image_for_closest_pixel = rc.camera.get_depth_image()
 
    # Crop off the ground directly in front of the car
    cropped_image = rc_utils.crop(depth_image_for_closest_pixel, (0, 0), (int(rc.camera.get_height() * 0.6), rc.camera.get_width()))
 
    # Find the closest pixel
    closest_pixel = rc_utils.get_closest_pixel(cropped_image)
    distanceFromObj = cropped_image[closest_pixel[0], closest_pixel[1]] 
    #print(distanceFromObj)
 
    ### DEPTH CAMERA - FIND CLOSEST PIXEL ###
 
 
 
 
    ### SAFETY STOP ###
 
    # Stop the car acording to the distance of the object
    if shouldDeductSpeed and not hax:
        if distanceFromObj < 100.9 and distanceFromObj > 60.9 and not override and counter < 2.1:
            hax = True
        if distanceFromObj < 100.9 and distanceFromObj > 60.9 and not override:
            setSpeedTo0 = False
            isAbleToDrive = True
            speed -= 0.5
        elif distanceFromObj <= 60.8 and distanceFromObj >= 50.9 and not override:
            isAbleToDrive = True
            speed -= 0.8
        elif distanceFromObj <= 50.8 and distanceFromObj >= 21.9 and not override:
            isAbleToDrive = False
            speed = -0.5
        elif distanceFromObj <= 21.8 and not override:
            shouldDeductSpeed = False
            rc.drive.stop()
    elif not setSpeedTo0:
        speed = 0
        angle = 0
        setSpeedTo0 = True
 
    ### SAFETY STOP ###
 
 
 
 
    ### SLOPE ###
    
    counter += rc.get_delta_time()
 
    print(counter)
 
    ### SLOPE ###
 
 
 
 
    # Use the left joystick to control the angle of the front wheels
    angle = rc.controller.get_joystick(rc.controller.Joystick.LEFT)[0]
 
    # Clamp speed before being applied to avoid errors
    speed = rc_utils.clamp(speed, -1, 1)
    rc.drive.set_speed_angle(speed, angle)
 
    # Print the current speed and angle when the A button is held down
    if rc.controller.is_down(rc.controller.Button.A):
        print("Speed:", speed, "Angle:", angle)
 
    # Print the depth image center distance when the B button is held down
    if rc.controller.is_down(rc.controller.Button.B):
        print("Center distance:", center_distance)
 
    # Display the cropped depth image
    rc.display.show_depth_image(cropped_image)
 
    # TODO (stretch goal): Prevent forward movement if the car is about to drive off a
    # ledge.  ONLY TEST THIS IN THE SIMULATION, DO NOT TEST THIS WITH A REAL CAR.
 
    # TODO (stretch goal): Tune safety stop so that the car is still able to drive up
    # and down gentle ramps.
    # Hint: You may need to check distance at multiple points.
 
 
########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################
 
if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()

