import cv2
import numpy as np
import imutils
import sys

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
        f = cv2.flip(frame,1)
        kernel = np.ones((6, 6), np.uint8)
        hsv_frame = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
        lhsv = np.array([5, 175, 80])
        uhsv = np.array([255, 255, 255])

        # create mask based on hsv range
        mask = cv2.inRange(hsv_frame, lhsv, uhsv)
        clos_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


        # Find contours:
        contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        final = cv2.bitwise_and(f,f,mask=clos_mask)
        fin= final
        #f0 = cv2.add(f0,final)
        # Draw contours:
        X,Y,c = 0,0,0
        points = []

        f0 = cv2.add(f0,fin)
        #cv2.imshow('mask',clos_mask)
        cv2.imshow("feed", f)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #'''
    cv2.imshow("finalimg",f0)
    mf = cv2.inRange(cv2.cvtColor(f0,cv2.COLOR_RGB2HSV),lhsv,uhsv)
    cv2.imshow("finalmask",mf)
    cv2.imwrite("alph.png",mf)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #'''
c=1
flag = 0
while True:
    if flag ==1:
        break
    cap = cv2.VideoCapture(0)
    while True:
        _,f = cap.read()
        f= cv2.flip(f,1)
        cv2.imshow("press p to draw", f)
        if cv2.waitKey(1) & 0xFF == ord('p'):
            cv2.destroyWindow("press p to draw")
            cap.release()
            capture()
            i = resize()
            s = "alph" + str(c) + ".png"
            cv2.imwrite(s, i)
            c += 1
            break
        if cv2.waitKey(1) & 0xFF == 27:
            flag=1
            break

cap.release()
cv2.destroyAllWindows()