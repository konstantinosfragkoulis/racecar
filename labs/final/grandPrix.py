import sys
import cv2 as cv
import numpy as np

sys.path.insert(1, "../../library")
import racecar_core
import racecar_utils as rc_utils



rc = racecar_core.create_racecar()

CROP_FLOOR = ((360, 0), (rc.camera.get_height(), rc.camera.get_width()))

protentialColors = [
    ((40, 50, 50), (80, 255, 255), "green"),
    ((170, 50, 50), (10, 255, 255), "red"),
    ((130, 50, 50), (160, 255, 255), "purple"),
    ((10, 50, 50), (20, 255, 255), "orange")
]

ORANGE = ((10, 150, 150), (20, 255, 255))
PURPLE = ((130, 100, 100), (160, 255, 255))
GREEN = ((40, 50, 50), (80, 255, 255))

speed = 0.0
angle = 0.0
timeElapsed = 0.0

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



def start():
    rc.drive.stop()

def update():

    global speed
    global angle
    global contourCenter

    updateContour()
    #setMaxSpeed()
    rc.drive.set_max_speed(0.3)

    if contourCenter is not None:
        angle = rc_utils.remap_range(contourCenter[1], 0, 640, -1, 1, True)

    speed = rc_utils.clamp(1 - angle, -1, 1)

    rc.drive.set_speed_angle(speed, angle)



if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
