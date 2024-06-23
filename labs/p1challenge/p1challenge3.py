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
from enum import IntEnum

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils
import pidcontroller
import filterOnePole

########################################################################################
# Global variables
########################################################################################


class State(IntEnum):
    Go_To_Cone = 1
    Blue_Cone = 2
    Red_Cone = 3
    Bailout = 4
    Stop = 5
    RecoverConeLeft = 6
    RecoverConeRight = 7


# Add any global variables here
rc = racecar_core.create_racecar()

# HSV ranges
BLUE = ((100, 150, 50), (125, 255, 255))
RED = ((170, 84, 50), (10, 255, 255))

MIN_CONTOUR = 30
# BAILOUT_DIST = 20
# BAILOUT_CENTER_THRESHHOLD = 80
DIST_CENTER = 200
MIN_DIST = 15
CUTOFF_FREQUENCY = 10
START_TURN = 40
# MAX_CONE_DIST = 80

# >> Variables
speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels
counter = 0
currentState = State.Stop
width = 0
height = 0
LowpassFilter = None
Angle_PID = pidcontroller.PID(3, 0.1, 1)
oldDist = 0
oldColor = False
########################################################################################
# Functions
########################################################################################


def getContourDist(contour, depth_image):
    mask = np.zeros_like(depth_image, np.uint8)
    mask = cv.drawContours(mask, [contour], 0, (255), -1)
    d = np.copy(depth_image)
    d[mask == 0] = 0
    return np.mean(np.ma.masked_equal(d, 0))


def getCone(color_image, depth_image):
    """Return cone center, distance, color"""
    color = False  # False = blue, True = Red
    if color_image is None or depth_image is None:
        contour_center = None
        dist = 0
    else:
        # Find all of the orange contours
        contours_Blue = rc_utils.find_contours(color_image, BLUE[0], BLUE[1])
        contours_Red = rc_utils.find_contours(color_image, RED[0], RED[1])

        min_dist = 0
        contour = None

        for c in contours_Red:
            if cv.contourArea(c) > MIN_CONTOUR:
                dist = getContourDist(c, depth_image)
                if dist < min_dist or min_dist == 0:
                    min_dist = dist
                    contour = c
                    color = True

        for c in contours_Blue:
            if cv.contourArea(c) > MIN_CONTOUR:
                dist = getContourDist(c, depth_image)
                if dist < min_dist or min_dist == 0:
                    min_dist = dist
                    contour = c
                    color = False

        dist = min_dist
        if contour is not None:
            # Calculate contour information
            # contour_center = rc_utils.get_contour_center(contour)

            # points = np.argwhere(mask > 0)
            # print(points)
            retval, triangle = cv.minEnclosingTriangle(contour)

            i = np.argmin(triangle[:, 0, 1])

            contour_center = np.copy(triangle[i, 0])
            triangle = triangle.flatten()

            # print(triangle)

            # Draw contour onto the image
            rc_utils.draw_contour(color_image, contour)
            # rc_utils.draw_circle(color_image, contour_center)
            # draw triangle
            cv.line(
                color_image,
                (triangle[0], triangle[1]),
                (triangle[2], triangle[3]),
                (255, 0, 255),
                2,
            )
            cv.line(
                color_image,
                (triangle[0], triangle[1]),
                (triangle[4], triangle[5]),
                (0, 255, 0),
                2,
            )
            cv.line(
                color_image,
                (triangle[2], triangle[3]),
                (triangle[4], triangle[5]),
                (0, 255, 255),
                2,
            )
        else:
            contour_center = None
            dist = 0

        # Display the image to the screen
        # rc.display.show_color_image(color_image)
        # rc.display.show_depth_image(depth_image)

    return contour_center, dist, color


def start():
    """
    This function is run once every time the start button is pressed
    """
    # Have the car begin at a stop
    rc.drive.stop()

    global LowpassFilter
    LowpassFilter = filterOnePole.Filter(
        filterOnePole.Type.LOWPASS, CUTOFF_FREQUENCY, 0, False
    )

    global counter
    counter = 0
    global currentState
    currentState = State.Stop

    global width
    global height
    width = rc.camera.get_width()
    height = rc.camera.get_height()

    global oldDist
    global oldCenter
    oldDist = 0
    oldCenter = 0

    # Print start message
    print(">> Phase 1 Challenge: Cone Slaloming")


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    color_image = rc.camera.get_color_image()
    depth_image = rc.camera.get_depth_image()

    contour_center, coneDist, color = getCone(
        color_image, depth_image
    )  # contour center is x, y

    global currentState
    global counter
    global speed
    global angle
    global oldDist
    global oldColor

    if np.ma.is_masked(coneDist):
        coneDist = oldDist

    if coneDist != 0:
        coneDist = LowpassFilter.input(coneDist, rc.get_delta_time())
        # print(contour_center[1], " ", coneDist, "red" if color else "blue")

    target = -1

    # FSM
    if currentState == State.Go_To_Cone:
        if contour_center is None:
            if oldDist > START_TURN:  # lost the cone, not ready to turn
                if oldColor:
                    print("lost cone red, recover")
                    currentState = State.RecoverConeLeft
                else:
                    print("lost cone blue, recover")
                    currentState = State.RecoverConeRight
            else:  # lost the cone, ready to turn
                if oldColor:
                    print("lost cone red, turn")
                    currentState = State.Red_Cone
                else:
                    print("lost cone blue, turn")
                    currentState = State.Blue_Cone
            # currentState = State.Stop
        else:
            if coneDist > DIST_CENTER:
                target = width // 2
            elif color:  # red
                target = rc_utils.remap_range(
                    coneDist, START_TURN, DIST_CENTER, 10, width - 10
                )
            else:  # blue
                target = rc_utils.remap_range(
                    coneDist, START_TURN, DIST_CENTER, width - 10, 10
                )
            angleError = (contour_center[0] - target) / (width // 2)
            if coneDist < START_TURN and angleError < 0.1:
                counter = 0
                if color:
                    currentState = State.Red_Cone
                else:
                    currentState = State.Blue_Cone
            angle = rc_utils.clamp(
                Angle_PID.update(angleError, rc.get_delta_time()), -1, 1
            )
            speed = 1
    if currentState == State.RecoverConeRight:
        speed = 0.2
        angle = 1
        if contour_center is not None:
            if coneDist < MIN_DIST:
                currentState = State.Bailout
            else:
                currentState = State.Go_To_Cone
    if currentState == State.RecoverConeLeft:
        speed = 0.2
        angle = -1
        if contour_center is not None:
            if coneDist < MIN_DIST:
                currentState = State.Bailout
            else:
                currentState = State.Go_To_Cone
    if currentState == State.Red_Cone:
        if contour_center is not None and coneDist < MIN_DIST:
            currentState = State.Bailout
        elif counter < 0.5:  # autonomous portion
            print("red autonomous 1")
            speed = 1
            angle = 0.4
        elif counter < 1:  # autonomous portion
            print("red autonomous 2")
            speed = 0.6
            angle = -1
        else:  # guided portion
            """"speed = 0.3
            target = width // 2
            angleError = (contour_center[0] - target) / (width // 2)
            angle = np.clip(Angle_PID.update(angleError, rc.get_delta_time()), -1, 1)
            if angleError < 0.1:"""
            currentState = State.Go_To_Cone
    if currentState == State.Blue_Cone:
        if contour_center is not None and coneDist < MIN_DIST:
            currentState = State.Bailout
        if counter < 0.5:  # autonomous portion
            print("blue autonomous 1")
            speed = 1
            angle = -0.4
        elif counter < 1:  # autonomous portion
            print("blue autonomous 2")
            speed = 0.6
            angle = 1
        else:  # guided portion
            """speed = 0.3
            target = width // 2
            angleError = (contour_center[0] - target) / (width // 2)
            angle = np.clip(Angle_PID.update(angleError, rc.get_delta_time()), -1, 1)
            if angleError < 0.1:"""
            currentState = State.Go_To_Cone
    if currentState == State.Stop:
        speed = 0
        angle = 0
        if contour_center is not None and rc.controller.was_pressed(
            rc.controller.Button.A
        ):
            currentState = State.Go_To_Cone
    if currentState == State.Bailout:  # move backwards for 2 seconds
        if counter < 2:
            speed = -1
            angle = 0
        elif counter < 3:
            speed = 0
            angle = 0
        else:
            currentState = State.Go_To_Cone

    if target > -1:
        cv.line(
            color_image, (int(target), 0), (int(target), height), (0, 255, 255), 2,
        )

    rc.display.show_color_image(color_image)

    counter += rc.get_delta_time()
    rc.drive.set_speed_angle(speed, angle)

    if contour_center is not None and abs(oldDist - coneDist) < 20:
        oldColor = color
    oldDist = coneDist

    # TODO: Slalom between red and blue cones.  The car should pass to the right of
    # each red cone and the left of each blue cone.


def update_slow():
    """
    After start() is run, this function is run at a constant rate that is slower
    than update().  By default, update_slow() is run once per second
    """
    # Print a line of ascii text denoting the steering angle
    s = ["-"] * 33
    s[16 + int(angle * 16)] = "|"
    print(
        "".join(s) + " : speed = " + str(speed),
        " angle = " + str(angle),
        " State:",
        currentState,
    )


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
