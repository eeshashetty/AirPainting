import cv2
import numpy as np
import imutils
import sys
import keras
from keras.models import load_model

model = load_model('emnist_model.h5')


def resize():
    img = cv2.imread("alph.png")
    newimg = imutils.resize(img, height=28)
    w = newimg.shape[1]
    newimg = newimg[0:28, int((w - 28) / 2):int((w + 28) / 2)]
    newimg = cv2.cvtColor(newimg,cv2.COLOR_RGB2GRAY)
    return newimg



def capture():
    cap = cv2.VideoCapture(0)
    while True:
        _, f0 = cap.read()
        if cv2.waitKey(0):
            break

    while True:
        _, frame = cap.read()
        f = cv2.flip(frame, 1)

        kernel = np.ones((6, 6), np.uint8)
        hsv_frame = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)

        lhsv = np.array([10, 100, 20])
        uhsv = np.array([255, 255, 255])

        # create mask based on hsv range
        mask = cv2.inRange(hsv_frame, lhsv, uhsv)
        clos_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours:
        (contours, _) = cv2.findContours(clos_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            if radius > 10:
                cv2.circle(f, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                M = cv2.moments(c)
                try:
                    center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                    points.append(center)
                except:
                    pass

        for i in range(1, len(points)):
            if points[i - 1] is None or points[i] is None:
                continue
            cv2.line(f, points[i - 1], points[i], (0, 255, 0), 5)
            cv2.line(f0, points[i - 1], points[i], (255, 255, 255), 5)

        cv2.imshow("feed (press q after drawing)", f)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #'''
    cv2.imshow("final (press q to exit)",f0)
    cv2.imwrite("alph.png",f0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #'''
c=1
flag = 0
while True:
    if flag==1:
        break
    cap = cv2.VideoCapture(0)
    while True:
        _,f = cap.read()
        f= cv2.flip(f,1)
        cv2.imshow("press d to draw (esc to exit)", f)
        if cv2.waitKey(1) & 0xFF == ord('d'):
            cv2.destroyWindow("press d to draw (esc to exit)")
            cap.release()
            points = []
            capture()
            i = resize()
            s = "alph" + str(c) + ".png"
            cv2.imwrite(s, i)
            c += 1
            i = i.reshape(-1,28,28,1)
            model.predict(i)
            break
        if cv2.waitKey(1) & 0xFF == 27:
           flag = 1
           cap.release()
           break
       
        
cap = cv2.VideoCapture(0)

while True:
    _, f0 = cap.read()
    if cv2.waitKey(0):
        break

while True:
    _, frame = cap.read()
    f = cv2.flip(frame, 1)

    kernel = np.ones((6, 6), np.uint8)
    hsv_frame = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
	print('lol ur mom gay')
    lhsv = np.array([6, 120, 20])
    uhsv = np.array([255, 255, 255])

    # create mask based on hsv range
    mask = cv2.inRange(hsv_frame, lhsv, uhsv)
    clos_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('mask',clos_mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
