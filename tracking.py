# REFERENCES
# - Masking color: https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
# - Contour features: https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
# - Bounding box: https://stackoverflow.com/questions/23398926/drawing-bounding-box-around-given-size-area-contour
# - HSV RGB conversion: https://stackoverflow.com/questions/24852345/hsv-to-rgb-color-conversion
#
# IMPROVEMENTS
# I used openCV's contour tutorial and made it so that my webcam could track the largest blue object 
# with a minimum enclosing box. I also added a flag to toggle between HSV and RGB

import numpy as np
from matplotlib.colors import hsv_to_rgb
import cv2

cap = cv2.VideoCapture(0)

# True for HSV, False for RGB
color_toggle = True

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come her
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # define range of blue color in HSV
    lower_blue_hsv = np.array([110,50,50])
    upper_blue_hsv = np.array([130,255,255])

    lower_blue_rgb = hsv_to_rgb(lower_blue_hsv)
    upper_blue_rgb = hsv_to_rgb(upper_blue_hsv)
    
    # Threshold the HSV image to get only blue colors
    mask = None
    if (color_toggle):
        mask = cv2.inRange(hsv, lower_blue_hsv, upper_blue_hsv)
    else:
        mask = cv2.inRange(rgb, lower_blue_rgb, upper_blue_rgb)

    contours,junk=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    frame_max_area = 0
    max_x, max_y, max_w, max_h = 0,0,0,0

    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        
        if (w * h > frame_max_area):
            frame_max_area = w * h
            max_x = x
            max_y = y
            max_w = w
            max_h = h

    cv2.rectangle(frame,(max_x,max_y),(max_x+max_w,max_y+max_h),(0,255,0),2)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    cv2.imshow('frame',frame)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()