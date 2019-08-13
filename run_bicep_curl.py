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


logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


def euclidian(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def angle_calc(p0, p1, p2):
    '''
        p1 is center point from where we measured angle between p0 and p2
    '''
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=str, default=0)
    parser.add_argument('--resize', type=str, default='432x368',
                        help='if provided, resize images before they are processed. default=432x368, '
                             'Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
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
        ret_val, image = cam.read()
        i = 1
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
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

        # This is the part of the loop that recognizes whether a bicep curl is being performed or not
        if len(pose) > 0:

            # set up text to speech module
            engine = pyttsx3.init()
            voices = engine.getProperty("voices")
            engine.setProperty("rate", 165)
            engine.setProperty("voice", "english-us")

            if state == "start":
                correctArm = find_correct_arm(human)
                correctArmv2 = find_correct_arm_v2(human)
                # if correctArm == "Both arms detected, further analysis required..":
                    # correctArm = correctArmv2
                        # if correctArm is "Both arms detected, further analysis required.." \
                        # else "Can't identify correct arm."

            if prevCorrectArm is not correctArm:
                print('Exercise arm detected as: {}.'.format(correctArm))
                prevCorrectArm = correctArm

            angleCorrectArm, prevArmAngle = find_correct_arm_angle(angleCorrectArm, correctArm)

            # if i don't include the stuff before continue the video will freeze for a couple milliseconds
            # before resuming
            if angleCorrectArm == 0 or abs(angleCorrectArm - prevArmAngle) > 40:
                cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
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

                if frm == 0:
                    out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                          (width, height))
                    # print("Initializing")
                    frm += 1
                cv2.imshow('tf-pose-estimation result', image)
                if i != 0:
                    out.write(image)
                fps_time = time.time()
                if cv2.waitKey(1) == 27:
                    break

            if angleCorrectArm > maxArmAngle and state is "bicep_curl_top" or state is "bicep_curl_up" or state is\
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


                # print(state)
                # print("elbow positioning: ({}, {})".format(x_elbow, y_elbow))
                # print("hip positioning: ({}, {})".format(x_hip, y_hip))

            # bicep_curl_start -> start
            elif angleCorrectArm > 166 and state == "bicep_curl_start":
                prevState = state
                state = "start"
                instructions = "You returned to starting position and restarted exercise. Please follow instructions" \
                               "to correctly complete exercise."
                badForm = False
                # print(state)

            # bicep_curl_start -> bicep_curling_up
            elif 145 > angleCorrectArm > 65 and state == "bicep_curl_start":
                prevState = state
                state = "bicep_curling_up"
                instructions = "Continue raising hand towards shoulder"
                command_line = "python /home/coles/tf-pose-estimation/subprocess_audio_scripts/curl_up.py"
                args1 = shlex.split(command_line)
                subprocess.Popen(args1)
                badForm = False
                print("Curl Up")
                # print(state)

            # bicep_curling_up -> bicep_curling_top or bicep_curling_up -> bicep_curl_finish
            elif angleCorrectArm < 65 and state == "bicep_curling_up":
                prevState = state
                state = "bicep_curl_top"
                instructions = "Once you feel you can't raise your hand further towards your shoulder, reverse your " \
                               "previous motion and while keeping upper arm parallel to torso, slowly lower your hand" \
                               "back towards your hip"
                down = True
                up = False
                command_line = "python /home/coles/tf-pose-estimation/subprocess_audio_scripts/curl_down.py"
                args1 = shlex.split(command_line)
                subprocess.Popen(args1)
                badForm = False
                print("Curl Down")
                # print(state)
                update_Min_Max_values(x_elbow, y_elbow, elbow_Min_max)
                update_Min_Max_values(x_hip, y_hip, hip_Min_Max)
                # print("elbow positioning: ({}, {})".format(x_elbow, y_elbow))
                # print("hip positioning: ({}, {})".format(x_hip, y_hip))

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
                # print(state)
                update_Min_Max_values(x_elbow, y_elbow, elbow_Min_max)
                update_Min_Max_values(x_hip, y_hip, hip_Min_Max)
                # print("elbow positioning: ({}, {})".format(x_elbow, y_elbow))
                # print("hip positioning: ({}, {})".format(x_hip, y_hip))

            # bicep_curl_top -> bicep_curling_down
            elif 145 > angleCorrectArm > 65 and down and state == "bicep_curl_top":
                prevState = state
                state = "bicep_curling_down"
                instructions = "Continue lowering hand towards hip"
                badForm = False
                # print(state)

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
                # print(state)

            elif 166 > angleCorrectArm > 145 and state == "bicep_curling_down":
                prevState = state
                state = "bicep_curl_finish"
                instructions = "Congratulations! You have successfully completed a bicep curl!"
                update_Min_Max_values(x_elbow, y_elbow, elbow_Min_max)
                update_Min_Max_values(x_hip, y_hip, hip_Min_Max)
                aveXHip = (hip_Min_Max[1] + hip_Min_Max[0]) / 2
                maxElbowDist = abs(elbow_Min_max[1] - aveXHip)
                minElbowDist = abs(elbow_Min_max[0] - aveXHip)
                # print("aveXHip: ", aveXHip)
                # print("maxElbowDist: ", maxElbowDist)
                # print("minElbowDist: ", minElbowDist)
                bicepCount += 1
                if maxElbowDist > 45 or minElbowDist > 45:
                    feedback = "Your upper arm shows significant rotation around the shoulder when curling." \
                               " Try holding your upper arm still, parallel to your chest, and concentrate on " \
                               "rotating around your elbow only."
                    command_line = "python /home/coles/tf-pose-estimation/subprocess_audio_scripts/success_bad_form.py {}".format(bicepCount)
                    args1 = shlex.split(command_line)
                    subprocess.Popen(args1)
                    badForm = True
                else:
                    feedback = "You have completed this exercise with good form! Weight was lifted fully up and" \
                               " upper arm/shoulder did not move significantly."
                    command_line = "python /home/coles/tf-pose-estimation/subprocess_audio_scripts/success_good_form.py {}".format(bicepCount)
                    args1 = shlex.split(command_line)
                    subprocess.Popen(args1)
                    badFormCount = 0
                    badForm = False
                # print(state)
                    # print("elbow positioning: ({}, {})".format(x_elbow, y_elbow))
                    # print("hip positioning: ({}, {})".format(x_hip, y_hip))
                print(feedback)

            # bicep_curl_finish -> start or bicep_curl_finish -> bicep_curling_up
            elif state == "bicep_curl_finish":
                state = "start"
                instructions = ""
                elbow_Min_max = [100000, -1, 100000, -1]
                hip_Min_Max = [100000, -1, 100000, -1]
                # print(feedback)
                # badForm = False
                badFormCount = 0
                # print(state)

            cv2.putText(image, "Curling: {}".format(state), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), thickness=2)
            cv2.putText(image,
                        "Instructions: ", ((inPos - 120), 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)
            cv2.putText(image, "{} Arm angle: {}".format(correctArm, angleCorrectArm), (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)
            cv2.putText(image, "Rep Number: {}".format(bicepCount), (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)

            if badForm and badFormCount < 25:
                cv2.putText(image, "BAD FORM!", (int(width / 2), int(height * (1/8))), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255),thickness=2)
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

                cv2.putText(image, line, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2,
                    lineType = cv2.LINE_AA) # (int(width/2) - 150, 110)

                j += 1

        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if frm == 0:
            out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                              (width, height))
            # print("Initializing")
            frm += 1
        cv2.imshow('tf-pose-estimation result', image)
        if i != 0:
            out.write(image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()
