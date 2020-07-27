#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import pickle
from datetime import datetime


# In[2]:


#load cascade classifier training file for haarcascade
xml_path = 'C:/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(xml_path)


# In[8]:


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./face-trainner.yml")


# In[9]:


labels = {}
with open("face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

labels


# In[17]:


# Function to detect and recognize faces in a frame

def detect_face_vid(frame):
    
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        
    if len(faces):
        if ts[2] == False:
            ts[0] = datetime.now()
            ts[2] = True

        for (x,y,w,h) in faces:
            #Draw rectangle around faces
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            #Try to recognize face
            roi_gray = gray[y:y+h, x:x+w] 
            id_, conf = recognizer.predict(roi_gray)
            #print(conf, id_)

            #if conf>=50 and conf<=95:
            font = cv2.FONT_HERSHEY_SIMPLEX 
            name = str(labels[id_])
            cv2.putText(frame, name, (x,y), font, 1, (255, 255, 255), 2) 
            ts[3] = name
    else:
        if ts[2] == True:
            ts[1] = datetime.now()
            ts[2] = False
            diff = ts[1]-ts[0]
            if diff.seconds!=0:
                faceFound.append((ts[3], diff.seconds, ts[0].strftime("%m/%d/%Y %H:%M:%S"), ts[1].strftime("%m/%d/%Y %H:%M:%S")))
            #faceFound[labels[id_]] = diff.seconds
            #print('Face Detected. From:',ts[0].strftime("%m/%d/%Y, %H:%M:%S"),' to:',ts[1].strftime("%m/%d/%Y, %H:%M:%S"))
        pass        
        
    return frame


# ### For Video

# In[18]:


# Capture time duration of faces detected
faceFound = []
ts = [0,0,False,'Unknown']

video_capture = cv2.VideoCapture("C:/Users/Administrator/Desktop/FlipRobo/FRP03-FaceRecognition/facerecog3.mp4")
#video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()
    if ret:
        frame_with_face=detect_face_vid(cv2.flip(cv2.transpose(frame), 1))
        cv2.imshow("Result",frame_with_face)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
video_capture.release()
cv2.destroyAllWindows()

print(faceFound)


# In[ ]:




