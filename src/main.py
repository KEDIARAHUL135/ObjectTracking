import cv2
import numpy as np
import src.macros as M
import time

t1 = time.time()

global cap
if M.VIDEO_OR_WEBCAM == 1:
    cap = cv2.VideoCapture(M.INPUT_VIDEO_PATH)
else:
    cap = cv2.VideoCapture(0)

_, OldFrame = cap.read()


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

    Frame = NewFrame.copy()

    DiffFrame = cv2.absdiff(OldFrame, NewFrame)
    BlurDiffFrame = cv2.GaussianBlur(DiffFrame, (5, 5), 2)
    GrayDiffFrame = cv2.cvtColor(BlurDiffFrame, cv2.COLOR_BGR2GRAY)
    _, ThreshDiffFrame = cv2.threshold(GrayDiffFrame, 20, 255, cv2.THRESH_BINARY)
    DilateDiffFrame = cv2.dilate(ThreshDiffFrame, (11, 11), iterations=10)

    Contours, _ = cv2.findContours(DilateDiffFrame, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

    for Contour in Contours:
        if cv2.contourArea(Contour) < M.MIN_CONTOUR_AREA:
            continue

        (x, y, w, h) = cv2.boundingRect(Contour)
        cv2.rectangle(Frame, (x, y), (x+w, y+h), (0, 0, 255), thickness=2)

    cv2.drawContours(Frame, Contours, -1, (0, 255, 0), thickness=2)

    DisplayFrames(NewFrame, DiffFrame, ThreshDiffFrame, DilateDiffFrame, Frame)
    OldFrame = NewFrame.copy()

    Key = cv2.waitKey(M.WAITKEY_VALUE)
    if Key == 32:       # Break when space is pressed
        break

cv2.destroyAllWindows()
cap.release()
t2 = time.time()
print(t2-t1)