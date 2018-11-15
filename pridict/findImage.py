import cv2
import numpy as np
import os
import glob

files = glob.glob('/home/namvh/Documents/do an may hoc/file/pridict/image/*')
for f in files:
    os.remove(f)
files = glob.glob('/home/namvh/Documents/do an may hoc/file/pridict/feature/*')
for f in files:
    os.remove(f)

cascadePath = "/home/namvh/Documents/do an may hoc/file/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
im = cv2.imread("/home/namvh/Documents/do an may hoc/file/imageTest/test4.jpg")
gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
sampleNum = 0   
for(x,y,w,h) in faces:
    sampleNum = sampleNum + 1
    cv2.imwrite("/home/namvh/Documents/do an may hoc/file/pridict/image/img." + '1.' + str(sampleNum) + ".jpg", im[y:y + h, x:x + w])
