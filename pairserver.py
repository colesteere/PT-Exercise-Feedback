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
        # print("Difference in left side (hand - hip): ", diffLeft)
        # print("Difference in right side (hand - hip): ", diffRight)
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
fps_time = 0
# w, h = model_wh("432x368")
w, h = model_wh("256x256")
# e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(432, 368))
e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(256, 256))

shoulder_Min_max = [100000, -1, 100000, -1]  # [xMin, xMax, yMin, yMax]
correctHip = "No hip detected"
prevCorrectHip = ""
angleCorrectHip = -1
prevHipAngle = -1
abductorCount = 0
prevHipCoords = (-1, -1)
correctHipCoords = (-1, -1)


while True:

    print("Receiving frame... ")
    data = socket.recv()
    np_data = np.fromstring(data, np.uint8)
    image = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

    print("Processing image... ")

    # turn the encoded data back into an image and then do all the processing that you need to do on it.
    i = 1
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
    pose = humans
    human = find_correct_human(humans)
    image = TfPoseEstimator.draw_humans(image, human, imgcopy=False)
    height, width = image.shape[0], image.shape[1]
    isCurling = False
    inPos = int(width * .75)
    shoulderPosition = False
    y = 0
    correctTorsoVector = -1
    correctArmVector = -1
    x_elbow = -1
    y_elbow = -1
    x_hip = -1
    y_hip = -1
    x_shoulder = -1
    y_shoulder = -1

    # This is the part of the loop that recognizes whether a bicep curl is being performed or not
    if bicep_curl:
        if len(pose) > 0:
            if state == "start":
                correctArm = find_correct_arm(human)
                correctArmv2 = find_correct_arm_v2(human)

            if prevCorrectArm is not correctArm:
                print('Exercise arm detected as: {}.'.format(correctArm))
                prevCorrectArm = correctArm

            angleCorrectArm, prevArmAngle = find_correct_arm_angle(angleCorrectArm, correctArm)

            # if i don't include the stuff before continue the video will freeze for a couple milliseconds
            # before resuming
            if angleCorrectArm == 0 or abs(angleCorrectArm - prevArmAngle) > 40:
                cv2.putText(image, "Rep Number: {}".format(bicepCount), (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)
                cv2.putText(image, "Instructions: ", ((inPos - 120), 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), thickness=2)
                cv2.putText(image, "{} Arm angle:".format(correctArm), (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)
                cv2.putText(image, "Curling: {}".format(state), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), thickness=2)
                if state is "bicep_curl_start" or state is "bicep_curling_up":
                    cv2.arrowedLine(image, (inPos, 140), (inPos, 80), (0, 0, 255), 2, 8, 0, 0.3)
                    if spokeUp is False:
                        spokeUp = True
                        spokeDown = False
                elif state is "bicep_curl_top" or state is "bicep_curling_down":
                    cv2.arrowedLine(image, (inPos, 80), (inPos, 140), (0, 0, 255), 2, 8, 0, 0.3)
                    if spokeDown is False:
                        spokeDown = True
                        spokeUp = False

            if angleCorrectArm > maxArmAngle and state is "bicep_curl_top" or state is "bicep_curl_up" or state is \
                    "bicep_curl_start":
                maxArmAngle = angleCorrectArm
            if angleCorrectArm < minArmAngle and state is "bicep_curl_start":
                minArmAngle = angleCorrectArm

            if state is "bicep_curl_start" or state is "bicep_curling_up":
                cv2.arrowedLine(image, (inPos, 140), (inPos, 80), (0, 0, 255), 2, 8, 0, 0.3)
                if spokeUp is False:
                    spokeUp = True
                    spokeDown = False
            elif state is "bicep_curl_top" or state is "bicep_curling_down":
                cv2.arrowedLine(image, (inPos, 80), (inPos, 140), (0, 0, 255), 2, 8, 0, 0.3)
                if spokeDown is False:
                    spokeDown = True
                    spokeUp = False

            try:
                elbow = human.body_parts[3 if correctArm == 'right' else 6]  # elbow point
                hip = human.body_parts[8 if correctArm == 'right' else 11]
                x_elbow = elbow.x * width
                y_elbow = elbow.y * height
                x_hip = hip.x * width
                y_hip = hip.y * width
            except:
                pass

            # start -> bicep_curl_start
            if 166 > angleCorrectArm > 145 and state == "start" and correctArm != "Both arms detected, further analysis required..":
                prevState = state
                state = "bicep_curl_start"
                instructions = "While keeping upper arm parallel to torso, " \
                               "contract bicep to raise hand towards shoulder"
                update_Min_Max_values(x_elbow, y_elbow, elbow_Min_max)
                update_Min_Max_values(x_hip, y_hip, hip_Min_Max)
                badForm = False

                print(state)

            # bicep_curl_start -> start
            elif angleCorrectArm > 166 and state == "bicep_curl_start":
                prevState = state
                state = "start"
                instructions = "You returned to starting position and restarted exercise. Please follow instructions" \
                               "to correctly complete exercise."
                badForm = False
                print(state)

            # bicep_curl_start -> bicep_curling_up
            elif 145 > angleCorrectArm > 65 and state == "bicep_curl_start":
                prevState = state
                state = "bicep_curling_up"
                instructions = "Continue raising hand towards shoulder"
                badForm = False
                print("Curl Up")
                print(state)

            # bicep_curling_up -> bicep_curling_top or bicep_curling_up -> bicep_curl_finish
            elif angleCorrectArm < 65 and state == "bicep_curling_up":
                prevState = state
                state = "bicep_curl_top"
                instructions = "Once you feel you can't raise your hand further towards your shoulder, reverse your " \
                               "previous motion and while keeping upper arm parallel to torso, slowly lower your hand" \
                               "back towards your hip"
                down = True
                up = False
                badForm = False
                print("Curl Down")
                print(state)
                update_Min_Max_values(x_elbow, y_elbow, elbow_Min_max)
                update_Min_Max_values(x_hip, y_hip, hip_Min_Max)

            elif angleCorrectArm < 60 and state == "bicep_curling_down":
                prevState = state
                state = "bicep_curl_top"
                badForm = True
                if prevState == "bicep_curling_down":
                    diffArmAngle = 30
                    feedback = "You have not lowered the weight all the way down to the original position. Remember " \
                               "to lower the weight all the way down to the point your forearm is parallel with your " \
                               "torso in order to have correct form and get the best results."
                    print(feedback)
                down = True
                up = False
                print(state)
                update_Min_Max_values(x_elbow, y_elbow, elbow_Min_max)
                update_Min_Max_values(x_hip, y_hip, hip_Min_Max)

            # bicep_curl_top -> bicep_curling_down
            elif 145 > angleCorrectArm > 65 and down and state == "bicep_curl_top":
                prevState = state
                state = "bicep_curling_down"
                instructions = "Continue lowering hand towards hip"
                badForm = False
                print(state)

            # bicep_curling_down -> bicep_curl_top or bicep_curling_down -> bicep_curl_finish
            elif 166 > angleCorrectArm > 145 and state == "bicep_curling_up":
                prevState = state
                state = "bicep_curl_finish"
                instructions = "Bad form, please try again."
                if prevState == "bicep_curling_up":
                    feedback = "You are not curling the weight all the way to the top, up to your shoulders." \
                               " Try to curl your arm completely so that your forearm is parallel with your torso." \
                               " It may help to use lighter weight."
                    print(feedback)
                badForm = True
                print(state)

            elif 166 > angleCorrectArm > 145 and state == "bicep_curling_down":
                prevState = state
                state = "bicep_curl_finish"
                instructions = "Congratulations! You have successfully completed a bicep curl!"
                update_Min_Max_values(x_elbow, y_elbow, elbow_Min_max)
                update_Min_Max_values(x_hip, y_hip, hip_Min_Max)
                aveXHip = (hip_Min_Max[1] + hip_Min_Max[0]) / 2
                maxElbowDist = abs(elbow_Min_max[1] - aveXHip)
                minElbowDist = abs(elbow_Min_max[0] - aveXHip)
                bicepCount += 1
                if maxElbowDist > 45 or minElbowDist > 45:
                    feedback = "Your upper arm shows significant rotation around the shoulder when curling." \
                               " Try holding your upper arm still, parallel to your chest, and concentrate on " \
                               "rotating around your elbow only."
                    badForm = True
                else:
                    feedback = "You have completed this exercise with good form! Weight was lifted fully up and" \
                               " upper arm/shoulder did not move significantly."
                    badFormCount = 0
                    badForm = False
                print(state)
                print(feedback)

            # bicep_curl_finish -> start or bicep_curl_finish -> bicep_curling_up
            elif state == "bicep_curl_finish":
                state = "start"
                instructions = ""
                elbow_Min_max = [100000, -1, 100000, -1]
                hip_Min_Max = [100000, -1, 100000, -1]
                # print(feedback)
                badForm = False
                badFormCount = 0
                print(state)

            cv2.putText(image, "Curling: {}".format(state), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), thickness=2)
            cv2.putText(image,
                        "Instructions: ", ((inPos - 120), 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)
            cv2.putText(image, "{} Arm angle: {}".format(correctArm, angleCorrectArm), (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)
            cv2.putText(image, "Rep Number: {}".format(bicepCount), (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)

            if badForm and badFormCount < 25:
                cv2.putText(image, "BAD FORM!", (int(width / 2), int(height * (1 / 8))), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), thickness=2)
                badFormCount += 1

            if badFormCount == 25:
                badFormCount = 0

        else:
            state = "start"
            prevState = "start"
            spokeUp = False
            spokeDown = True
            bicepCount = 0
            elbow_Min_max = [100000, -1, 100000, -1]
            hip_Min_Max = [100000, -1, 100000, -1]
            text = "Please stand with curling arm facing camera. Please ensure the upper half of your body can be " \
                   "seen by the camera."
            wrapped_text = textwrap.wrap(text, width=35)
            j = 0
            for line in wrapped_text:
                textsize = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                gap = textsize[1] + 10

                y = int((image.shape[0] + textsize[1]) / 2 - 110) + j * gap
                x = int((image.shape[1] - textsize[0]) / 2)

                cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2,
                            lineType=cv2.LINE_AA)  # (int(width/2) - 150, 110)

                j += 1

        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        fps_time = time.time()

    elif hip_abductor:
        if len(pose) > 0:
            rightHipAngle, leftHipAngle = find_hip_angle()
            print("Left Hip Angle: ", leftHipAngle)
            print("Right Hip Angle: ", rightHipAngle)

            # if not right and not left:
            if state == "start":
                correctHip = find_correct_hip(rightHipAngle, leftHipAngle)

            if (prevCorrectHip is not correctHip) and state == "start":
                prevState = state
                state = "start"
                if correctHip != "Can't identify correct hip.":
                    print('{} hip detected.'.format(correctHip))
                else:
                    print('{}'.format(correctHip))
                prevCorrectHip = correctHip

            if correctHip == "Right":
                prevHipCoords = correctHipCoords
                correctHipCoords = (find_point(human, 9)[0], find_point(human, 9)[1])
                if (state is "abductor_start" or state is "abductor_up") and up:
                    cv2.arrowedLine(image, (correctHipCoords[0] - 10, correctHipCoords[1] - 10),
                                    (correctHipCoords[0] - 70, correctHipCoords[1] - 70), (0, 0, 255), 2, 8, 0, 0.3)
                elif (state is "abductor_top" or state is "abductor_down") and down:
                    cv2.arrowedLine(image, (correctHipCoords[0] + 10, correctHipCoords[1] + 10),
                                    (correctHipCoords[0] + 70, correctHipCoords[1] + 70), (0, 0, 255), 2, 8, 0, 0.3)
            elif correctHip == "Left":
                prevHipCoords = correctHipCoords
                correctHipCoords = (find_point(human, 12)[0], find_point(human, 12)[1])
                print("Correct Hip Coordinates: ", correctHipCoords)
                if state is "abductor_start" or state is "abductor_up" and up:
                    cv2.arrowedLine(image, (correctHipCoords[0] + 10, correctHipCoords[1] - 10),
                                    (correctHipCoords[0] + 70, correctHipCoords[1] - 70), (0, 0, 255), 2, 8, 0, 0.3)
                elif state is "abductor_top" or state is "abductor_down" or state is "abductor_finished" and down:
                    if abs(prevHipCoords[1] - correctHipCoords[1]) > 50:
                        correctHipCoords[0] = prevHipCoords[0] - 10
                        correctHipCoords[1] = prevHipCoords[1] + 10
                    cv2.arrowedLine(image, (correctHipCoords[0] - 10, correctHipCoords[1] + 10),
                                    (correctHipCoords[0] - 70, correctHipCoords[1] + 70), (0, 0, 255), 2, 8, 0, 0.3)

            angleCorrectHip, prevHipAngle = find_correct_hip_angle(angleCorrectHip, correctHip)
            if prevHipAngle is not angleCorrectHip:
                prevHipAngle = angleCorrectHip

            if angleCorrectHip > 70 or abs(angleCorrectHip - prevHipAngle) > 40:
                cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            try:
                shoulder = human.body_parts[2 if correctHip == "Left" else 5]  # left hand point
                x_shoulder = shoulder.x * width
                y_shoulder = shoulder.y * height
            except:
                pass

            # Either left or right leg has been identified as the correct leg and exercise can start
            if state == "start":  # and correctHip != "Can't identify correct hip."
                prevState = state
                state = "abductor_start"
                shoulder_Min_max = update_Min_Max_values(x_shoulder, y_shoulder, shoulder_Min_max)
                up = True
                print(state)

            # abductor_start -> start
            elif angleCorrectHip < 11 and state == "abductor_start":
                prevState = state
                state = "start"
                print(state)

            # abductor_start -> abductor_up
            elif 40 > angleCorrectHip > 15 and state == "abductor_start":
                prevState = state
                state = "abductor_up"
                print(state)

            # abductor_up -> abductor_top
            elif angleCorrectHip > 40 and state == "abductor_up":
                prevState = state
                state = "abductor_top"
                up = False
                down = True
                update_Min_Max_values(x_shoulder, y_shoulder, shoulder_Min_max)
                print(state)

            # abductor_up -> abductor_finished
            elif angleCorrectHip < 15 and state == "abductor_up":
                prevState = state
                state = "abductor_finished"
                up = False
                down = False
                feedback = "You did not raise your leg to the top of your range of motion. Please try and lift your" \
                           " leg as high as you can and try again."
                print(state)
                print(feedback)

            # abductor_top -> abductor_down
            elif 40 > angleCorrectHip > 20 and state == "abductor_top":
                prevState = state
                state = "abductor_down"
                is_lean_too_far(shoulder_Min_max)
                print(state)

            # abductor_down -> abductor_top
            elif angleCorrectHip > 45 and state == "abductor_down":
                prevState = state
                state = "abductor_top"
                feedback = "You did not lower your leg back to the bottom of your range of motion. Remember to lower " \
                           "your leg back to its starting position before starting another repetition."
                update_Min_Max_values(x_shoulder, y_shoulder, shoulder_Min_max)
                print(state)
                print(feedback)

            # abductor_down -> abductor_finished
            elif angleCorrectHip < 20 and state == "abductor_down":
                prevState = state
                state = "abductor_finished"
                abductorCount += 1
                up = False
                down = False
                update_Min_Max_values(x_shoulder, y_shoulder, shoulder_Min_max)
                lean = is_lean_too_far(shoulder_Min_max)
                if not lean:
                    feedback = "You have successfully completed a hip abuctor with good form. Next time try and lift " \
                               "your leg even higher than you did the last time."
                else:
                    feedback = "You have successfully completed a hip abductor, however during the exercise you were " \
                               "leaning too far to one side. Try and keep your torso stable and still as much as " \
                               "possible throughout the exercise."
                print(state)
                print(feedback)

            elif state == "abductor_finished":
                prevState = state
                state = "start"
                up = False
                down = False
                shoulder_Min_max = [100000, -1, 100000, -1]  # [xMin, xMax, yMin, yMax]
                print(state)

            cv2.putText(image, "Hip Abduction State: {}".format(state), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), thickness=2)
            cv2.putText(image, "Rep Number: {}".format(abductorCount), (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)
            if correctHip != "Can't identify correct hip.":
                cv2.putText(image, "{} Hip angle: {}".format(correctHip, angleCorrectHip), (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)
            else:
                cv2.putText(image, "{}".format(correctHip), (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)

        else:
            abductorCount = 0
            shoulder_Min_max = [100000, -1, 100000, -1]  # [xMin, xMax, yMin, yMax]
            text = "Please stand facing camera with your entire body in full view of the camera."
            wrapped_text = textwrap.wrap(text, width=35)
            j = 0
            for line in wrapped_text:
                textsize = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                gap = textsize[1] + 10

                y = int((image.shape[0] + textsize[1]) / 2 - 110) + j * gap
                x = int((image.shape[1] - textsize[0]) / 2)

                cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=2,
                            lineType=cv2.LINE_AA)  # (int(width/2) - 150, 110)

                j += 1

        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        fps_time = time.time()

    print ("Finished processing image, sending processed image back to the client...")
    _, newdata = cv2.imencode('.jpg', image)
    socket.send(newdata.tostring())
