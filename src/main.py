###############################################################################
# File          : main.py
# Created by    : Rahul Kedia
# Created on    : 19/01/2020
# Project       : ObjectTracking
# Description   : This file contains the main source code for the project.
################################################################################

import cv2
import src.macros as M
import time

t1 = time.time()

# Creating object for reading video as input
global cap
if M.VIDEO_OR_WEBCAM == 1:
    cap = cv2.VideoCapture(M.INPUT_VIDEO_PATH)
else:
    cap = cv2.VideoCapture(0)

_, OldFrame = cap.read()


################################################################################
# Function      : DisplayFrames
# Parameter     : {All the parameters are clearly mentioned and contains
#                 the images which are to be shown.}
# Description   : This function shows images as required.
# Return        : -
################################################################################
def DisplayFrames(NewFrame, DiffFrame, ThreshDiffFrame, DilateDiffFrame, Frame):

    if M.SHOW_ALLorOUTPUT == 1:
        cv2.imshow("CurrentFrame", NewFrame)
        cv2.imshow("DiffFrame", DiffFrame)
        cv2.imshow("ThressDiffFrame", ThreshDiffFrame)
        cv2.imshow("DilateDiffFrame", DilateDiffFrame)

    cv2.imshow("Contours", Frame)


while cap.isOpened():
    ret, NewFrame = cap.read()

    if ret is False:
        break

    # Copying frame to print output on it. This is done so that frame to be processed is not disturbed.
    Frame = NewFrame.copy()

    # Implementing different operations to get image with white area in which movement has occured.
    DiffFrame = cv2.absdiff(OldFrame, NewFrame)
    BlurDiffFrame = cv2.GaussianBlur(DiffFrame, (5, 5), 2)
    GrayDiffFrame = cv2.cvtColor(BlurDiffFrame, cv2.COLOR_BGR2GRAY)
    _, ThreshDiffFrame = cv2.threshold(GrayDiffFrame, M.THRESHOLD_AT, 255, cv2.THRESH_BINARY)
    DilateDiffFrame = cv2.dilate(ThreshDiffFrame, (11, 11), iterations=M.DILATION_ITERATIONS)

    # Finding contours of movement
    Contours, _ = cv2.findContours(DilateDiffFrame, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

    for Contour in Contours:
        # too small contours of movement/noise is rejected.
        if cv2.contourArea(Contour) < M.MIN_CONTOUR_AREA:
            continue

        # Creating rectangle bounding the contour. (Red)
        (x, y, w, h) = cv2.boundingRect(Contour)
        cv2.rectangle(Frame, (x, y), (x+w, y+h), (0, 0, 255), thickness=2)

    # Drawing contours (Green)
    cv2.drawContours(Frame, Contours, -1, (0, 255, 0), thickness=2)

    # Function for displaying frames
    DisplayFrames(NewFrame, DiffFrame, ThreshDiffFrame, DilateDiffFrame, Frame)

    # Copying current frame to old frame for next iteration/set of frames.
    OldFrame = NewFrame.copy()

    # Escape when space bar is hit.
    Key = cv2.waitKey(M.WAITKEY_VALUE)
    if Key == 32:       # Break when space is pressed
        break

cv2.destroyAllWindows()
cap.release()
t2 = time.time()
print(t2-t1)