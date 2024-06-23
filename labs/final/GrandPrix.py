"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Grand Prix
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
import pidcontroller
import transformations as ts
import filterOnePole

#######################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

Wall_Angle_PID = pidcontroller.PID(4, 0.5, 0.5)  # d = 0.5
Angle_Onscreen_PID = pidcontroller.PID(3, 0.5, 0.1)
width = rc.camera.get_width()
height = rc.camera.get_height()
old_challenge = None
cur_challenge = None
targetFilter = filterOnePole.Filter(filterOnePole.Type.LOWPASS, 1, width // 2)

ts.num_samples = rc.lidar.get_samples()
ts.VIS_RADIUS = 300
ts.width = width

# Add any global variables here
speed = 0.0
angle = 0.0
counter = 0
cone_count = 0

MIN_CONTOUR = 30
MIN_CONTOUR_AREA = 700
TURN_THRESHOLD = 70
CROP_FLOOR = ((360, 0), (rc.camera.get_height(), rc.camera.get_width()))


class Color(IntEnum):
    Red = 0
    Blue = 1
    Green = 2
    Orange = 3
    Purple = 4
    White = 5
    Yellow = 6
    # Red1 = 7
    # Red2 = 8


class Challenge(IntEnum):
    Line = 1
    Lane = 2
    Cones = 3
    # Slalom = 3
    # Gate = 4
    Wall = 4
    ManualControl = 5


class ConeState(IntEnum):
    DRIVE = 0
    RED = 1
    BLUE = 2


cur_conestate = ConeState.DRIVE

HSV_RANGE = [None for i in range(len(Color))]
# Colors, stored as a pair (hsv_min, hsv_max)
HSV_RANGE[Color.Blue] = ((80, 120, 100), (125, 255, 255))
HSV_RANGE[Color.Green] = ((45, 120, 150), (75, 255, 255))
HSV_RANGE[Color.Red] = ((170, 84, 100), (180, 255, 255), (0, 84, 100), (10, 255, 255))
HSV_RANGE[Color.Purple] = ((125, 90, 100), (140, 255, 255))
HSV_RANGE[Color.Orange] = ((10, 95, 150), (25, 255, 255))

BGR = [None for i in range(len(Color))]
BGR[Color.Red] = (0, 0, 255)
BGR[Color.Blue] = (255, 127, 0)
BGR[Color.Green] = (0, 255, 0)
BGR[Color.Orange] = (0, 127, 255)
BGR[Color.Purple] = (255, 0, 127)
BGR[Color.White] = (255, 255, 255)
BGR[Color.Yellow] = (0, 255, 255)

########################################################################################
# Functions
########################################################################################


def manualControl():
    rt = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
    lt = rc.controller.get_trigger(rc.controller.Trigger.LEFT)
    speed = rt - lt
    angle = rc.controller.get_joystick(rc.controller.Joystick.LEFT)[0]
    return speed, angle


def smooth(scan):
    temp = np.copy(scan)

    for i in range(rc.lidar.get_num_samples()):
        if temp[i] == 0 or temp[i] > MAX_LIDAR_RANGE:
            temp[i] = temp[i - 1]

    for i in range(-8, 8):
        if i != 0:
            temp += np.roll(scan, i)

    return temp / 17


def getWall(scan, startAngle, endAngle):
    # lidar_average_distance =rc_utils.get_lidar_average_distance(scan)
    scan_xy = ts.polar2TopDown(ts.lidar2Polar(scan, startAngle, endAngle))

    # scan_r[abs(scan_r - np.mean(scan_r)) > 2 * np.std(scan_r)] = 0

    # scan_xy = np.array(Vp2c(scan_r, scan_t))

    valid_points = np.all(scan_xy != 0, axis=0)
    x_points = scan_xy[0, valid_points]
    y_points = scan_xy[1, valid_points]

    scan_polynomial = np.poly1d(np.polyfit(x_points, y_points, 1))

    # data vis:

    l = Vc2p(np.arange(-100, 100), scan_polynomial(np.arange(-100, 100)))
    # l = Vc2p(x_points, y_points)

    l = np.transpose([np.degrees(l[1]), l[0]])
    # print(l)
    # s = np.zeros_like(scan)
    # s[startAngle * 2 : endAngle * 2] = scan[startAngle * 2 : endAngle * 2]
    # rc.display.show_lidar(s, radius=200, max_range=400, highlighted_samples=l)

    # y = mx + b
    # poly1d: [b, m]
    # print(scan_polynomial)
    distance = scan_polynomial(0)
    # distance = rc_utils.get_lidar_average_distance(
    #   scan, (startAngle + (endAngle - startAngle) / 2)
    # )
    angle = np.degrees(
        (np.pi / 2) - np.arctan(scan_polynomial[1])
    )  # arctan(slope = m) = angle
    return (distance, angle, l)


def getContourDist(contour, depth_image):
    mask = np.zeros_like(depth_image, np.uint8)
    mask = cv.drawContours(mask, [contour], 0, (255), -1)
    d = np.copy(depth_image)
    d[mask == 0] = 0
    return np.mean(np.ma.masked_equal(d, 0))


def getCone(color_image, depth_image):
    """Return cone center, distance, color"""
    color = None  # False = blue, True = Red
    if color_image is None or depth_image is None:
        contour_center = None
        dist = 0
    else:
        # Find all of the orange contours
        contours_Blue = rc_utils.find_contours(
            color_image, HSV_RANGE[Color.Blue][0], HSV_RANGE[Color.Blue][1]
        )
        contours_Red = rc_utils.find_contours(
            color_image, HSV_RANGE[Color.Red][0], HSV_RANGE[Color.Red][3]
        )

        min_dist = 0
        contour = None

        for c in contours_Red:
            if cv.contourArea(c) > MIN_CONTOUR:
                dist = getContourDist(c, depth_image)
                if dist < min_dist or min_dist == 0:
                    min_dist = dist
                    contour = c
                    color = Color.Red

        for c in contours_Blue:
            if cv.contourArea(c) > MIN_CONTOUR:
                dist = getContourDist(c, depth_image)
                if dist < min_dist or min_dist == 0:
                    min_dist = dist
                    contour = c
                    color = Color.Blue

        dist = min_dist
        if contour is not None:
            # Calculate contour information
            # contour_center = rc_utils.get_contour_center(contour)

            # points = np.argwhere(mask > 0)
            # print(points)
            retval, triangle = cv.minEnclosingTriangle(contour)

            i = np.argmin(triangle[:, 0, 1])

            contour_center = np.flip(triangle[i, 0])
            triangle = triangle.flatten()

            # Draw contour onto the image
            # rc_utils.draw_contour(color_image, contour)
            # rc_utils.draw_circle(color_image, contour_center)

            # draw triangle

            """cv.line(
                color_image,
                (triangle[0], triangle[1]),
                (triangle[2], triangle[3]),
                BGR[Color.Purple],
                2,
            )
            cv.line(
                color_image,
                (triangle[0], triangle[1]),
                (triangle[4], triangle[5]),
                BGR[Color.Purple],
                2,
            )
            cv.line(
                color_image,
                (triangle[2], triangle[3]),
                (triangle[4], triangle[5]),
                BGR[Color.Purple],
                2,
            )"""

        else:
            contour_center = None
            dist = 0
    return contour_center, dist, color


def update_contour(COLOR_PRIORITY, CROP, image):
    """
    Finds contours in the current color image and uses them to update contour_center
    and contour_area
    """
    contour_center = None
    global contour_area

    # image = rc.camera.get_color_image()

    if image is None:
        contour_center = None
        contour_area = 0
    else:
        # Crop the image to the floor directly in front of the car
        image = rc_utils.crop(image, CROP[0], CROP[1])
        current_color = None

        # Search for each color in priority order
        for color in COLOR_PRIORITY:
            # Find all of the contours of the current color
            if color == Color.Red:
                contours = rc_utils.find_contours(
                    image, HSV_RANGE[color][0], HSV_RANGE[color][3]
                )
            else:
                contours = rc_utils.find_contours(
                    image, HSV_RANGE[color][0], HSV_RANGE[color][1]
                )

            # Select the largest contour
            contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)

            if contour is not None:
                # Calculate contour information
                contour_center = rc_utils.get_contour_center(contour)
                contour_area = rc_utils.get_contour_area(contour)

                # Draw contour onto the image
                rc_utils.draw_contour(image, contour)
                rc_utils.draw_circle(image, contour_center)
                current_color = color
                break

        # If no contours are found for any color, set center and area accordingly
        else:
            contour_center = None
            contour_area = 0

        # Display the image to the screen
        # rc.display.show_color_image(image)
        return contour_center, current_color


def findAr(color_image, min_area):
    corners, ids = rc_utils.get_ar_markers(color_image)
    p1 = (corners[0][0][0], corners[0][0][1])
    p2 = (corners[0][1][0], corners[0][1][1])
    p3 = (corners[0][2][0], corners[0][2][1])
    p4 = (corners[0][3][0], corners[0][3][1])
    area = (p2[0] - p1[0]) * (p3[1] - p3[1])
    if area > min_area:
        return corners, ids
    return 0, 0


def drawLines(p, color_image, color):
    for line in p:
        cv.line(
            color_image, (int(line(height)), height), (int(line(0)), 0), BGR[color],
        )


def findIntersection(poly1, poly2):
    if poly1.c[0] == poly1.c[1]:
        return None
    # y = mx + b
    # Set both lines equal to find the intersection point in the x direction
    # m1 * x + b1 = m2 * x + b2
    # m1 * x - m2 * x = b2 - b1
    # x * (m1 - m2) = b2 - b1
    # x = (b2 - b1) / (m1 - m2)
    x = (poly2.c[1] - poly1.c[1]) / (poly1.c[0] - poly2.c[0])
    return x, poly1(x)


def start():
    global cur_challenge
    global cur_conestate
    global old_challenge
    global ar_counter
    global counter
    global cone_count
    counter = 0
    ar_counter = 0
    cone_count = 0
    cur_challenge = Challenge.ManualControl
    cur_conestate = ConeState.DRIVE
    old_challenge = Challenge.Wall
    """
    This function is run once every time the start button is pressed
    """
    # Have the car begin at a stop
    rc.drive.stop()
    rc.drive.set_max_speed(1)

    global width
    global height
    width = rc.camera.get_width()
    height = rc.camera.get_height()

    # Print start message
    print(">> Grand Prix")


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """

    scan = np.clip(rc.lidar.get_samples(), 0, None)  # smooth(rc.lidar.get_samples())

    scan_xy = None

    color_image = rc.camera.get_color_image()
    # depth_image = cv.bilateralFilter(rc.camera.get_depth_image(), 9, 75, 75)
    depth_image = rc.camera.get_depth_image()[::8, ::8]  # subsample for sim
    depth_image[depth_image == 0] = rc.camera.get_max_range()
    depth_image = cv.resize(
        depth_image, (width, height), interpolation=cv.INTER_LINEAR_EXACT
    )
    # vis_image = np.zeros((2 * VIS_RADIUS, 2 * VIS_RADIUS, 3), np.uint8, "C")
    hsv_image = cv.cvtColor(color_image, cv.COLOR_BGR2HSV)

    # FSM

    speed = 0
    angle = 0
    global cur_challenge
    global old_challenge
    global colorPriority
    global ar_counter
    if cur_challenge == Challenge.ManualControl:
        speed, angle = manualControl()
        if rc.controller.was_pressed(rc.controller.Button.A):
            cur_challenge = old_challenge
    else:
        if rc.controller.was_pressed(rc.controller.Button.A):
            old_challenge = cur_challenge
            cur_challenge = Challenge.ManualControl

    if cur_challenge == Challenge.Line:
        # Determine largest contour
        speed = 0.25
        contour_center, current_color = update_contour(
            [Color.Red, Color.Green, Color.Blue], CROP_FLOOR, color_image
        )
        error = (contour_center[1] - width // 2) / (width // 2)
        angle = rc_utils.clamp(
            Angle_Onscreen_PID.update(error, rc.get_delta_time()), -1, 1
        )
        if contour_center is None:
            cur_challenge = Challenge.Wall

    elif cur_challenge == Challenge.Lane:
        speed = 0.5
        angle = 0
        mask = cv.bitwise_or(
            cv.inRange(
                hsv_image, HSV_RANGE[Color.Purple][0], HSV_RANGE[Color.Purple][1]
            ),
            cv.inRange(
                hsv_image, HSV_RANGE[Color.Orange][0], HSV_RANGE[Color.Orange][1]
            ),
        )

        mask[: height // 2] = 0

        left_image = np.copy(mask)
        right_image = np.copy(mask)
        left_image[:, width // 2 :] = 0
        right_image[:, : width // 2] = 0

        lines = cv.HoughLinesP(
            left_image,
            rho=6,
            theta=np.pi / 60,
            threshold=200,
            # lines=np.array([]),
            minLineLength=40,
            maxLineGap=25,
        )

        left_lines = []

        if lines is not None:
            lines = lines[:, 0]
            for line in lines:  # line = [x1, y1, x2, y2]
                if line[3] != line[1]:
                    slope = float(line[2] - line[0]) / (line[3] - line[1])
                    if -5 < slope < 0:
                        left_lines.append(np.poly1d([slope, line[0] - slope * line[1]]))

        lines = cv.HoughLinesP(
            right_image,
            rho=6,
            theta=np.pi / 60,
            threshold=200,
            # lines=np.array([]),
            minLineLength=40,
            maxLineGap=25,
        )

        right_lines = []

        if lines is not None:
            lines = lines[:, 0]
            for line in lines:  # line = [x1, y1, x2, y2]
                if line[3] != line[1]:
                    slope = float(line[2] - line[0]) / (line[3] - line[1])
                    if 0 < slope:
                        right_lines.append(
                            np.poly1d([slope, line[0] - slope * line[1]])
                        )

        """if len(leftLines) >= 10:
            leftLines = leftLines[
                np.argsort(leftLines[:, 1])[
                    len(leftLines) // 2 - 5 : len(leftLines) // 2 + 5
                ]
            ]

        if len(rightLines) >= 10:
            rightLines = rightLines[
                np.argsort(rightLines[:, 1])[
                    len(rightLines) // 2 - 5 : len(rightLines) // 2 + 5
                ]
            ]"""

        if len(left_lines) == 0:
            if len(right_lines) == 0:
                print("No lines")
                # straight (default)
            else:
                print("No left lines")
                drawLines(right_lines, color_image, Color.Red)
                speed = 0.15
                angle = -1
                # hard left
        else:
            drawLines(left_lines, color_image, Color.Green)
            if len(right_lines) == 0:
                print("No right lines")
                speed = 0.15
                angle = 1
                # hard right
            else:
                # both
                drawLines(right_lines, color_image, Color.Red)

                points = []

                for line1 in left_lines:
                    for line2 in right_lines:
                        y, x = findIntersection(line1, line2)
                        points.append(x)

                points = np.array(points)
                points = points[np.isfinite(points)]
                points = points[abs(points - np.mean(points)) < 2 * np.std(points)]
                x = rc_utils.clamp(np.mean(points), 0, width - 1)
                if np.isfinite(x):
                    x = int(targetFilter.input(x, rc.get_delta_time()))
                    # print("Output: ", x)
                    color_image[:, x] = BGR[Color.Yellow]
                    error = (x - width // 2) / (width // 2)
                    angle = rc_utils.clamp(
                        Angle_Onscreen_PID.update(error, rc.get_delta_time()), -1, 1
                    )

    elif cur_challenge == Challenge.Cones:
        contour_center, contour_center_distance, color = getCone(
            color_image, depth_image
        )  # color: False = Blue, True = Red
        # failed to recognize cone: contour_center = None, contour_center_distance = None, color = False/True
        # corners, ids = rc_utils.get_ar_markers(color_image)
        # if ids is not None and cone_count == 10:
        #     angle = 0
        #     ar_counter += rc.get_delta_time()
        #     if ar_counter > 2:
        #         cur_challenge = Challenge.Line
        #         cone_count = 0
        angle = 0
        if cone_count == 10:
            cur_conestate = ConeState.TRANSITION
            if counter < 1.4:
                angle = 1
            elif counter > 4:
                cur_challenge = Challenge.Line
                cone_count = 0
            counter += rc.get_delta_time()

        speed = 0.25

        # States
        if cur_conestate == ConeState.DRIVE and cone_count < 10:
            if contour_center is not None:
                # rc_utils.draw_circle(color_image, contour_center)
                angle = rc_utils.clamp(
                    rc_utils.remap_range(
                        contour_center[1], 0, rc.camera.get_width(), -1, 1
                    ),
                    -1,
                    1,
                )
                if cone_count == 2:
                    angle = 0.85
                if 0 < contour_center_distance < TURN_THRESHOLD:
                    if color == Color.Red:
                        cur_conestate = ConeState.RED
                        counter = 0
                    else:
                        cur_conestate = ConeState.BLUE
                        counter = 0
        elif cur_conestate == ConeState.RED:
            if counter < 0.5:
                angle = 1
            elif contour_center is not None and color == Color.Blue:
                cur_conestate = ConeState.DRIVE
                cone_count += 1
            elif cone_count == 9 and (
                contour_center_distance > TURN_THRESHOLD or contour_center_distance == 0
            ):
                cone_count += 1
            else:
                angle = -1
            counter += rc.get_delta_time()
        elif cur_conestate == ConeState.BLUE:
            if counter < 0.5:
                angle = -1
            elif contour_center is not None and color == Color.Red:
                cur_conestate = ConeState.DRIVE
                cone_count += 1
            elif cone_count == 9 and (
                contour_center_distance > TURN_THRESHOLD or contour_center_distance == 0
            ):
                cone_count += 1
            else:
                angle = 1
            counter += rc.get_delta_time()

    elif cur_challenge == Challenge.Wall:
        # AR tag
        # corners = rc_utils.get_ar_markers(color_image)
        # if corners is not None:
        #    orientation = rc_utils.get_ar_direction(corners[0][0])
        #    if orientation == 3:  # Left
        #        if counter < 1:
        #            angle = -1
        #    elif orientation == 1:
        #        if counter < 1:
        #            angle = 1
        #    counter += rc.get_delta_time()

        rightWallDist, rightWallAngle, RightL = getWall(scan, 50, 130)
        leftWallDist, leftWallAngle, LeftL = getWall(scan, 250, 310)

        s = np.zeros_like(scan)

        s[50 * 2 : 130 * 2] = scan[50 * 2 : 130 * 2]
        s[250 * 2 : 310 * 2] = scan[250 * 2 : 310 * 2]

        rc.display.show_lidar(
            s,
            radius=200,
            max_range=200,
            highlighted_samples=np.concatenate([RightL, LeftL]),
        )

        speed = 0.25

        angleError = TARGET_ANGLE - (rightWallAngle + leftWallAngle) / 2
        distError = (rightWallDist + leftWallDist) / 2

        # if abs(angleError) < ANGLE_THRESHHOLD:
        #    error = distError
        # else:
        error = distError / 10 + np.sin(np.radians(angleError)) * 2  # angleError / 30

        angle = rc_utils.clamp(Wall_Angle_PID.update(error, rc.get_delta_time()), -1, 1)

        if getCone(color_image, depth_image) is None:
            cur_challenge = Challenge.Line
            counter = 0

    # rc.display.show_color_image(color_image)
    rc.drive.set_speed_angle(speed, angle)


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
