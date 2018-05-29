# -*- coding: utf-8 -*-
import cv2

smile_detect = cv2.CascadeClassifier('smile_api.xml')
face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect(gray , frame):
    faces=face_detect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        smile = smile_detect.detectMultiScale(roi_gray,1.7,23)
        for(smx,smy,smw,smh) in smile:
           cv2.rectangle(roi_color,(smx,smy),(smx+smw,smy+smh),(0,255,0),2)
           
    return frame
    
videoRec=cv2.VideoCapture(0)
while True:
    _,frame=videoRec.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas=detect(gray,frame)
    cv2.imshow('Video',canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;

videoRec.release()
cv2.destroyAllWindows()
