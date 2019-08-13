import argparse
import logging
import shlex
import subprocess
import textwrap
import time
from pprint import pprint
import cv2
import numpy as np
import sys
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import math
import pyttsx3

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


def find_point(human, p):  # (pose,p)
    # for point in pose:
    try:
        body_part = human.body_parts[p]
        return (int(body_part.x * width + 0.5), (int(body_part.y * height + 0.5)))
    except:
        return (0, 0)

    # return (0, 0)


def points_exists(human, p1, p2, p3):  # (humans, p1, p2, p3)
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


def find_correct_human(humans):
    correct_human = None
    max_human = -1
    for human in humans:
        tempSum = human.get_max_score()
        if tempSum > max_human:
            max_human = tempSum
            correct_human = human

    return correct_human


def update_Min_Max_values(x, y, bodyPartCoords):
    if x < 0 or y < 0:
        return bodyPartCoords
    newMinMax = bodyPartCoords
    if x < bodyPartCoords[0]:
        newMinMax[0] = x
    if x > bodyPartCoords[1]:
        newMinMax[1] = x
    if y < bodyPartCoords[2]:
        newMinMax[2] = x
    if y > bodyPartCoords[3]:
        newMinMax[3] = x
    return newMinMax


def is_lean_too_far(shoulder):
    lean = False

    x_diff = abs(shoulder[1] - shoulder[0])
    y_diff = abs(shoulder[3] - shoulder[2])
    # print(x_diff)
    # print(y_diff)
    if x_diff > 30 and y_diff > 30:
        lean = True

    return lean


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

    shoulder_Min_max = [100000, -1, 100000, -1]  # [xMin, xMax, yMin, yMax]
    global height, width
    average_ratio = 0
    down = False
    up = True
    state = "start"
    prevState = "start"
    human = None
    spokeUp = False
    spokeDown = True
    feedback = ""
    correctHip = "No hip detected"
    prevCorrectHip = ""
    angleCorrectHip = -1
    prevHipAngle = -1
    abductorCount = 0
    up = False
    down = False
    prevHipCoords = (-1,-1)
    correctHipCoords = (-1, -1)

    while True:
        ret_val, image = cam.read()
        i = 1
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        pose = humans
        human = find_correct_human(humans)
        image = TfPoseEstimator.draw_humans(image, human, imgcopy=False)
        height, width = image.shape[0], image.shape[1]
        inPos = int(width * .75)
        y = 0
        x_shoulder = -1
        y_shoulder = -1

        # This is the part of the loop that recognizes whether a hip abductor is being performed or not
        if len(pose) > 0:

            # set up text to speech module
            engine = pyttsx3.init()
            voices = engine.getProperty("voices")
            engine.setProperty("rate", 165)
            engine.setProperty("voice", "english-us")

            rightHipAngle, leftHipAngle = find_hip_angle()
            # print("Left Hip Angle: ", leftHipAngle)
            # print("Right Hip Angle: ", rightHipAngle)

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
                # print("Correct Hip Coordinates: ", correctHipCoords)
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
                continue

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
                # print(state)

            # abductor_start -> start
            elif angleCorrectHip < 11 and state == "abductor_start":
                prevState = state
                state = "start"
                # print(state)

            # abductor_start -> abductor_up
            elif 38 > angleCorrectHip > 15 and state == "abductor_start":
                prevState = state
                state = "abductor_up"
                command_line = "python /home/coles/tf-pose-estimation/subprocess_audio_scripts/lift_hip.py"
                args1 = shlex.split(command_line)
                subprocess.Popen(args1)
                # print(state)

            # abductor_up -> abductor_top
            elif angleCorrectHip > 38 and state == "abductor_up":
                prevState = state
                state = "abductor_top"
                up = False
                down = True
                update_Min_Max_values(x_shoulder, y_shoulder, shoulder_Min_max)
                command_line = "python /home/coles/tf-pose-estimation/subprocess_audio_scripts/lower_hip.py"
                args1 = shlex.split(command_line)
                subprocess.Popen(args1)
                # print(state)

            # abductor_up -> abductor_finished
            elif angleCorrectHip < 15 and state == "abductor_up":
                prevState = state
                state = "abductor_finished"
                up = False
                down = False
                feedback = "You did not raise your leg to the top of your range of motion. Please try and lift your" \
                           " leg as high as you can and try again."
                # print(state)
                print(feedback)

            # abductor_top -> abductor_down
            elif 38 > angleCorrectHip > 20 and state == "abductor_top":
                prevState = state
                state = "abductor_down"
                is_lean_too_far(shoulder_Min_max)
                # print(state)

            # abductor_down -> abductor_top
            elif angleCorrectHip > 45 and state == "abductor_down":
                prevState = state
                state = "abductor_top"
                feedback = "You did not lower your leg back to the bottom of your range of motion. Remember to lower " \
                           "your leg back to its starting position before starting another repetition."
                update_Min_Max_values(x_shoulder, y_shoulder, shoulder_Min_max)
                # print(state)
                print(feedback)

            # abductor_down -> abductor_finished
            elif angleCorrectHip < 20 and state == "abductor_down":
                prevState = state
                state = "abductor_finished"
                abductorCount += 1
                print("Rep Number: {}".format(abductorCount))
                up = False
                down = False
                update_Min_Max_values(x_shoulder, y_shoulder, shoulder_Min_max)
                lean = is_lean_too_far(shoulder_Min_max)
                if not lean:
                    feedback = "You have successfully completed a hip abuctor with good form. Next time try and lift " \
                               "your leg even higher than you did the last time."
                    command_line = "python /home/coles/tf-pose-estimation/subprocess_audio_scripts/success_bad_form.py {}".format(abductorCount)
                    args1 = shlex.split(command_line)
                    subprocess.Popen(args1)
                else:
                    feedback = "You have successfully completed a hip abductor, however during the exercise you were " \
                               "leaning too far to one side. Try and keep your torso stable and still as much as " \
                               "possible throughout the exercise."
                    command_line = "python /home/coles/tf-pose-estimation/subprocess_audio_scripts/success_good_form.py {}".format(abductorCount)
                    args1 = shlex.split(command_line)
                    subprocess.Popen(args1)
                # print(state)
                print(feedback)
                print('\n' * 2)


            elif state == "abductor_finished":
                prevState = state
                state = "start"
                up = False
                down = False
                shoulder_Min_max = [100000, -1, 100000, -1]  # [xMin, xMax, yMin, yMax]
                # print(state)

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
            if args.__getattribute__('camera') == 0:
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

