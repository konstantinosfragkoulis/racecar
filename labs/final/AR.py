"""
Final Challenge - Time Trial
"""

import sys
import cv2 as cv
import numpy as np

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils

rc = racecar_core.create_racecar()

potentialColors = [
    ((90, 50, 50), (120, 255, 255), "blue"),
    ((40, 50, 50), (80, 255, 255), "green"),
    ((170, 50, 50), (10, 255, 255), "red"),
    ((130, 100, 100), (160, 255, 255), "purple"),
    ((10, 50, 50), (20, 255, 255), "orange")
]

def start():
    rc.drive.stop()

def update():
    image = rc.camera.get_color_image()
    markers = rc_utils.get_ar_markers(image)

    for marker in markers:
        marker.detect_colors(image, potentialColors)
        print(marker)
        depthImg = rc.camera.get_depth_image()
        mkCorners = marker.get_corners()

        print("Marker Dist:", depthImg[mkCorners[0][0]][mkCorners[0][1]])

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
