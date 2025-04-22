# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:59:30 2025

@author: acer
"""

import cv2
import numpy as np
import sys

PREVIEW = 0
BLUR = 1
FEATURES = 2
CANNY = 3

feature_params = dict(maxCorners=500 , qualityLevel = 0.2 , minDistance = 15 , blockSize=9)

image_filter = PREVIEW

alive=True

win_name="Camera Filters"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
result = None


s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]
source = cv2.VideoCapture(s)

source.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)  # 4K Width
source.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160) # 4K Height
source.set(cv2.CAP_PROP_FPS, 60)  # Increase FPS if supported
source.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Use MJPEG format for better quality

# Confirm the settings
width = source.get(cv2.CAP_PROP_FRAME_WIDTH)
height = source.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = source.get(cv2.CAP_PROP_FPS)
print(f"Resolution: {int(width)}x{int(height)}, FPS: {fps}")


while alive:
    ret, frame = source.read()
    if not ret:
        break
    frame = cv2.flip(frame , 1)
    
    if image_filter == PREVIEW:
        result = frame
        
    elif image_filter == BLUR:
        result = cv2.blur(frame , (5,5))
        
    elif image_filter == FEATURES:
        result = frame
        frame_gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
        if corners is not None:
            for x, y in np.float32(corners).reshape(-1,2):
                cv2.circle(result, (int(x), int(y)), 10,(255,0,0),1)
                
    elif image_filter == CANNY:
        result = cv2.Canny(frame ,100 ,150)
        
    cv2.imshow(win_name,result)
    
    
    key = cv2.waitKey(1)
    if key == ord("Q") or key == ord("q") or key ==27:
        alive = False
    elif key == ord("C") or key == ord("c"):
        image_filter = CANNY
    elif key == ord("B") or key == ord("b"):
        image_filter = BLUR
    elif key == ord("F") or key == ord("f"):
        image_filter = FEATURES
    elif key == ord("P") or key == ord("p"):
        image_filter = PREVIEW
        
source.release()
cv2.destroyWindow(win_name)
cv2.destroyAllWindows()