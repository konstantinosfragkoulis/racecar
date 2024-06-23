"""
    Copyright MIT and Harvey Mudd College
  MIT License
   Summer 2020
  
   Lab 3B - Depth Camera Cone Parking
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
ORANGE = ((10, 50, 50), (20, 255, 255))
MIN_CONTOUR_AREA = 20
stopDriving = False
contour = None
########################################################################################
# Functions
########################################################################################
def updateContour():
    global contourCenter
    global contourArea
    image = rc.camera.get_color_image()
    if image is None:
        contourCenter = None
        contourArea = None
    else:
        contours = rc_utils.find_contours(image, ORANGE[0], ORANGE[1])
        contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)
    if contour is not None:
    # Calculate contour information
        contourCenter = rc_utils.get_contour_center(contour)
        contourArea = rc_utils.get_contour_area(contour)
        # Draw contour onto the image
        rc_utils.draw_contour(image, contour)
        rc_utils.draw_circle(image, contourCenter)
    else:
        contourCenter = None
        contourArea = None
        rc.display.show_color_image(image)

def start():
    """
    This function is run once every time the start button is pressed
    """
    # Have the car begin at a stop
    rc.drive.stop()
    # Print start message
    print(">> Lab 3B - Depth Camera Cone Parking")

def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    # TODO: Park the car 30 cm away from the closest orange cone.
    # Use both color and depth information to handle cones of multiple sizes.
    # You may wish to copy some of your code from lab2b.py
    global stopDriving

    # Find the contour of the cone and its center
    updateContour()
    # Capture depth image
    depthImage = rc.camera.get_depth_image()

    # Drive towards the cone
    if contourArea is not None and not stopDriving:
        if depthImage[contourCenter[0]][contourCenter[1]] == 30:
            rc.drive.stop()
            stopDriving = True
        elif depthImage[contourCenter[0]][contourCenter[1]] > 29.5:
            speed = 0.5
            angle = rc_utils.remap_range(contourCenter[1], 0, 639, -1, 1)
            rc.drive.set_speed_angle(speed, angle)
        elif depthImage[contourCenter[0]][contourCenter[1]] < 30.5:
            speed = -0.55
            angle = 0
            rc.drive.set_speed_angle(speed, angle)
########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
