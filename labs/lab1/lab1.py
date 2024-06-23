"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Lab 1 - Driving in Shapes
"""

########################################################################################
# Imports
########################################################################################

import sys
from typing import Counter

sys.path.insert(1, "../../library")
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

########################################################################################
# Functions
########################################################################################


def start():
    """
    This function is run once every time the start button is pressed
    """
    global speed
    global counter
    global isDrivingInCircle
    global isDrivingInSquare
    global isDrivingInFigureEight
    global isDrivingInLine

    speed = 0
    counter = 0
    isDrivingInCircle = False
    isDrivingInSquare = False
    isDrivingInFigureEight = False
    isDrivingInLine = False

    # Begin at a full stop
    rc.drive.stop()

    # Print start message
    print(
        ">> Lab 1 - Driving in Shapes\n"
        "\n"
        "Controls:\n"
        "    Right trigger = accelerate forward\n"
        "    Left trigger = accelerate backward\n"
        "    Left joystick = turn front wheels\n"
        "    A button = drive in a circle\n"
        "    B button = drive in a square\n"
        "    X button = drive in a figure eight\n"
        "    Y button = drive in a line\n"
    )



def update():

    global speed
    global counter
    global isDrivingInCircle
    global isDrivingInSquare
    global isDrivingInFigureEight
    global isDrivingInLine

    speed = 0

    

    (x, y) = rc.controller.get_joystick(rc.controller.Joystick.LEFT)

    if rc.controller.get_trigger(rc.controller.Trigger.RIGHT) > 0:
        speed = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
        rc.drive.set_speed_angle(speed, x)
    

    if rc.controller.get_trigger(rc.controller.Trigger.LEFT) > 0:
        speed = -rc.controller.get_trigger(rc.controller.Trigger.LEFT)
        rc.drive.set_speed_angle(speed, x)

    rc.drive.set_speed_angle(speed, x)



    if rc.controller.was_pressed(rc.controller.Button.A):
        isDrivingInCircle = True
        isDrivingInSquare = False
        isDrivingInFigureEight = False
        isDrivingInLine = False
        counter = 0
    

    if isDrivingInCircle:
        counter += rc.get_delta_time()

        if counter < 5.25:
            speed = 1
            x = 1
            rc.drive.set_speed_angle(speed, x)
        else:
            rc.drive.stop()
            isDrivingInCircle = False



    if rc.controller.was_pressed(rc.controller.Button.B):
        isDrivingInSquare = True
        isDrivingInCircle = False
        isDrivingInFigureEight = False
        isDrivingInLine = False
        counter = 0


    if isDrivingInSquare:
        counter += rc.get_delta_time()

        if counter < 1.5:
            rc.drive.set_speed_angle(1, 0)
        elif counter < 3.25:
            rc.drive.set_speed_angle(0.75, 1)
        elif counter < 4:
            rc.drive.set_speed_angle(1, 0)
        elif counter < 5.25:
            rc.drive.set_speed_angle(0.75, 1)
        elif counter < 6:
            rc.drive.set_speed_angle(1, 0)
        elif counter < 7.5:
            rc.drive.set_speed_angle(0.75, 1)
        elif counter < 8.5:
            rc.drive.set_speed_angle(1, 0)
        elif counter < 10:
            rc.drive.set_speed_angle(0.75, 1)
        elif counter < 11:
            rc.drive.set_speed_angle(-0.75, 0)
        else:
            rc.drive.stop()
        

    if rc.controller.was_pressed(rc.controller.Button.X):
        isDrivingInFigureEight = True
        isDrivingInCircle = False
        isDrivingInSquare = False
        isDrivingInLine = False
        counter = 0

    if isDrivingInFigureEight:
        counter += rc.get_delta_time()

        if counter < 3:
            rc.drive.set_speed_angle(1, 0)
        elif counter < 6.5:
            rc.drive.set_speed_angle(1, 1)
        elif counter < 8.5:
            rc.drive.set_speed_angle(1, 0)
        elif counter < 11.5:
            rc.drive.set_speed_angle(1, -1)
        elif counter < 12:
            rc.drive.set_speed_angle(-1, 0)
        else:
            rc.drive.stop()
        

    if rc.controller.was_pressed(rc.controller.Button.Y):
        isDrivingInLine = True
        isDrivingInCircle = False
        isDrivingInSquare = False
        isDrivingInFigureEight = True

    if isDrivingInLine:
        counter += rc.get_delta_time()
        if counter < 2:
            rc.drive.set_speed_angle(1, 0)
        else:
            rc.drive.stop()


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update)
    rc.go()
