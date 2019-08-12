import pickle
import socket
import select
from ww import f
import zmq
import argparse
import logging
import shlex
import time
from pprint import pprint
import cv2
import numpy as np
import sys
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import math
import pyttsx3
import textwrap
import subprocess

HEADERSIZE = 10
PORT = "1234"
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.bind("tcp://*:%s" % PORT)
# IP = "127.0.0.1"

'''
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((socket.gethostname(), PORT))
server_socket.listen(5)

'''

bicep_curl = True
hip_abductor = False

def euclidian(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def angle_calc(p0, p1, p2):
        # p1 is center point from where we measured angle between p0 and p2
    try:
        a = (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2
        b = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
        c = (p2[0] - p0[0]) ** 2 + (p2[1] - p0[1]) ** 2
        angle = math.acos((a + b - c) / math.sqrt(4 * a * b)) * 180 / math.pi
    except:
        return 0
    return int(angle)


def find_point(human, p): # (pose,p)
    #for point in pose:
    try:
        body_part = human.body_parts[p]
        return (int(body_part.x * width + 0.5), (int(body_part.y * height + 0.5)))
    except:
        return (0, 0)

    # return (0, 0)


def points_exists(human, p1, p2, p3): # (humans, p1, p2, p3)
    # for human in humans:
    sum = 0
    try:
        body_part1 = human.body_parts[p1]
        sum += 1
    except:
        # print("Body part not found")
        # return False
        pass
    try:
        body_part2 = human.body_parts[p2]
        sum += 1
    except:
        # print("Body part not found")
        # return False
        pass
    try:
        body_part3 = human.body_parts[p3]
        sum += 1
    except:
        # print("Body part not found")
        # return False
        pass
    return sum


def give_feedback(diffArmAngle, shoulderPosition, complete):
    feedback = ""
    if complete:
        return "You have completed this exercise with good form! Weight was lifted fully up and upper arm/shoulder" \
               "did not move significantly"
    if diffArmAngle > 70:
        feedback += "You are not curling the weight all the way to the top, up to your shoulders. Try to curl " \
                   "your arm completely so that your forearm is parallel with your torso. It may help to use " \
                   "lighter weight."
    if diffArmAngle < 50:
        feedback += "You are not lowering the weight all the way back to the starting position. Remember to lower the " \
                   "weight till your forearm is parallel with your torso in order to get full range of motion" \
                   "and the best results."
    if not shoulderPosition:
        feedback += "Your upper arm shows significant rotation around the shoulder when curling. Try holding your" \
                   " upper arm still, parallel to your chest, and concentrate on rotating around your elbow only."

    return feedback


def find_correct_arm(pose_seq):
    # TODO: add a condition that finds the correct arm when both arms have all points showing.
    right_present = points_exists(pose_seq, 4, 3, 2) # 1 if ... else 0
    left_present = points_exists(pose_seq, 7, 6, 5) # 1 if ... else 0
    right_count = right_present
    left_count = left_present
    if right_count == left_count:
        return "Both arms detected, further analysis required.."
    side = 'right' if right_count > left_count else 'left'

    return side


def find_correct_arm_v2(human):
    try:
        lhand = human.body_parts[7]  # left hand point
        rhand = human.body_parts[4]  # right hand point
        rhip = human.body_parts[8]
        lhip = human.body_parts[11]
        y_rhip = rhip.y * height
        y_lhip = lhip.y * height
        y_lhand = lhand.y * height
        y_rhand = rhand.y * height
        diffRight = abs(y_rhand - y_rhip)
        diffLeft = abs(y_lhand - y_lhip)
        print("Difference in left side (hand - hip): ", diffLeft)
        print("Difference in right side (hand - hip): ", diffRight)
        if diffRight > 19:
            return "Right"
        elif diffLeft > 19:
            return "Left"
        return "Can't identify correct arm."
    except:
        pass


def find_correct_arm_angle(angleCorrectArm, correctArm):
    prevArmAngle = -1
    if correctArm == 'right':
        prevArmAngle = angleCorrectArm
        angleCorrectArm = angle_calc(find_point(human, 4), find_point(human, 3), find_point(human, 2))  # right arm

    elif correctArm == 'left':
        prevArmAngle = angleCorrectArm
        angleCorrectArm = angle_calc(find_point(human, 7), find_point(human, 6), find_point(human, 5))  # left arm

    return angleCorrectArm, prevArmAngle


def find_correct_human(humans):
    correct_human = None
    max_human = -1
    for human in humans:
        tempSum = human.get_max_score()
        if tempSum > max_human:
            max_human = tempSum
            correct_human = human

    return correct_human


def shoulder_movement(max, min):
    if max[-2] - min[-2] > 20:
        return True
    return False


def update_Min_Max_values(x, y, bodyPart):
    if x < 0 or y < 0:
        return bodyPart
    newMinMax = bodyPart
    if x < bodyPart[0]:
        newMinMax[0] = x
    if x > bodyPart[1]:
        newMinMax[1] = x
    if y < bodyPart[2]:
        newMinMax[2] = y
    if y > bodyPart[3]:
        newMinMax[3] = y
    return newMinMax


def find_correct_hip(rightAngle, leftAngle):

    if rightAngle > 11:
        return 'Right'
    elif leftAngle > 11:
        return 'Left'

    return "Can't identify correct hip."


def find_hip_angle():
    rightAngle = angle_calc(find_point(human, 9), find_point(human, 8), find_point(human, 11)) - 90 # right hip
    leftAngle = angle_calc(find_point(human, 12), find_point(human, 11), find_point(human, 8)) - 90 # left hip

    return rightAngle, leftAngle


def find_correct_hip_angle(angleCorrectHip, correctHip):
    prevHipAngle = -1
    if correctHip == "Right":
        prevHipAngle = angleCorrectHip
        angleCorrectHip = angle_calc(find_point(human, 9), find_point(human, 8), find_point(human, 11)) - 90 # right hip

    elif correctHip == "Left":
        prevHipAngle = angleCorrectHip
        angleCorrectHip = angle_calc(find_point(human, 12), find_point(human, 11), find_point(human, 8)) - 90 # left hip

    elif correctHip == "Can't identify correct hip.":
        prevHipAngle = -1
        angleCorrectHip = -1

    return angleCorrectHip, prevHipAngle


def is_lean_too_far(shoulder):
    lean = False

    x_diff = abs(shoulder[1] - shoulder[0])
    y_diff = abs(shoulder[3] - shoulder[2])
    # print(x_diff)
    # print(y_diff)
    if x_diff > 30 and y_diff > 30:
        lean = True

    return lean

count = 0
i = 0
frm = 0
elbow_Min_max = [100000, -1, 100000, -1]  # [xMin, xMax, yMin, yMax]
hip_Min_Max = [100000, -1, 100000, -1]  # [xMin, xMax, yMin, yMax]
global height, width
average_ratio = 0
bicepCount = 0
correctArm = "No arm detected"
prevCorrectArm = ""
maxArmAngle = -1
minArmAngle = 361
down = False
up = True
state = "start"
prevState = "start"
human = None
spokeUp = False
spokeDown = True
feedback = ""
angleCorrectArm = -1
prevArmAngle = -1
badForm = False
badFormCount = 0



while True:

    data = b''

    try:
        r = socket.recv(90456)
        if len(r) == 0:
            exit(0)
        a = r.find(b'END!')
        if a != -1:
            data += r[:a]
            break
        data += r
    except Exception as e:
        print (e)
        continue

    print("Processing image... ")

    # turn the encoded data back into an image and then do all the processing that you need to do on it.
    np_data = np.fromstring(data, np.uint8)
    image = cv2.imdecode(np_data, cv2.IMREAD_COLOR) # bgr_img
    mirror_img = cv2.flip(image, 1)
    _, jpeg_img = cv2.imencode('.jpg', image)

    print ("Finished processing image, sending processed image back to the client...")
    socket.send(jpeg_img)






    # this should be where all of the image processing happens






    '''
    clientsocket, address = server_socket.accept()
    print("Connection from {} has been established.".format(address))

    d = {1: "Hey", 2: "There"}
    msg = pickle.dumps(d, 2)

    msg = bytes(f('{len(msg):<{HEADERSIZE}}'), 'utf-8') + msg

    clientsocket.send(msg)
    '''