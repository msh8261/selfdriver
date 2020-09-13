import os
import gym
import gym_donkeycar
import eventlet.wsgi

import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
import matplotlib.pyplot as plt




def canny_func(image):
    # convert RGB 3 channels image to gray 1 channel image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #apply filter with 5x5 kernel here to reduce noise
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    # apply canny to find the edge of object it get derivative of image
    # if there is small change of pixel value means small change of d
    #erivative and vice versa
    low_threshold = 50
    high_threshold = 150
    canny = cv2.Canny(blur, low_threshold, high_threshold)
    return canny


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[[0, height-30], [320, height-30], [150, 60]]],  dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    # here we use and operation od mask and image to only keep the important edge
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image



def display_lines(image, lines):
    line_image = np.zeros_like(image)
    line_thickness=10
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1,y1), (x2, y2), (255,0,0), line_thickness)

    return line_image



def make_coordinates(image, line_parameters):
    ## y = mx+b ==> slope is m and intercept is b
    ## x = (y-b)/m
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1- intercept)/slope)
    x2 = int((y2- intercept)/slope)

    return np.array([x1, y1, x2, y2])




def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2),(y1, y2),1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))


    if(len(lines)<2):
        pass

    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)



    return np.array([left_line, right_line])




def run_on_image(image):
    # copy image to prevent changing of original image
    lane_image = np.copy(image)

    canny = canny_func(image)
    plt.imshow(canny)

    ROI = region_of_interest(canny)

    ## show by plot to find the number to find the roi values
    # plt.imshow(ROI)
    # plt.show()

    # threshold for hough means number of point to make a line
    thr = 50

    lines = cv2.HoughLinesP(canny, 2, np.pi/180, thr, np.array([]), minLineLength=40, maxLineGap=5)


    #if lines is not  None:

    average_lines = average_slope_intercept(lane_image, lines)
        # print("================", average_lines)

        # line_image = display_lines(lane_image, average_lines)
        #
        # combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
        #
        # cv2.imshow("win", combo_image)
        #
        # cv2.waitKey(1)

    return average_lines





def pid_controller(y, yc, h=1, Ti=1, Td=1, Kp=1, u0=0, e0=0):
    # Step variable
    k = 0
    # Initialization
    ui_prev = u0
    e_prev = e0
    while 1:
        # Error between the desired and actual output
        e = yc - y

        # Integration Input
        ui = ui_prev + 1.0 / Ti * h * e
        # Derivation Input
        ud = 1.0 / Td * (e - e_prev) / float( h)

        # Adjust previous values
        e_prev = e
        ui_prev = ui

        # Calculate input for the system
        u = Kp * (e + ui + ud)

        k += 1

        return u




def my_controller(slope1, slope2):
    if (slope1 >= -0.5 and slope2 <= 0.5):
        u = 0
    elif (slope1 < -0.5):
        u = 0.1 * (np.abs(slope1 / 0.5))
    elif (slope2 > 0.5):
        u = -0.1 * (np.abs(slope2 / 0.5))

    return u





sio = socketio.Server()

app = Flask(__name__)  # '__main__'

speed_limit = 15


def img_preprocess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img


@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image0 = np.copy(image)

    # change as 'pid' or 'dl' (for deep learning model)
    mode = 'pid'

    if(mode=='pid'):

        lines = run_on_image(image0)
        x01, y01, x02, y02 = lines[0].reshape(4)
        x11, y11, x12, y12 = lines[1].reshape(4)
        parameters0 = np.polyfit((x01, x02), (y01, y02), 1)
        parameters1 = np.polyfit((x11, x12), (y11, y12), 1)
        slope1 = parameters0[0]
        slope2 = parameters1[0]

        y = slope1 + slope2
        yc = 0

        u = pid_controller(y, yc, 1, 1, 1, 1, 0, 0)

        #u = my_controller(slope1, slope2)

        steering_angle = float(u)
        throttle = (1.0 - speed / speed_limit)
        print('{} {} {}'.format(steering_angle, throttle, speed))
        send_control(steering_angle, throttle)

    if (mode == 'dl'):
        image = img_preprocess(image)
        image = np.array([image])
        steering_angle = float(model.predict(image))
        throttle = 1.0 - speed / speed_limit
        print('{} {} {}'.format(steering_angle, throttle, speed))
        send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })


if __name__ == '__main__':
    # model = load_model('model/model2.h5')
    model = load_model('../../model/model3.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
