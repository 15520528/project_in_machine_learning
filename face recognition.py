import cv2,os
import numpy as np
from PIL import Image
cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('/home/namvh/Documents/do an cuoi cung/file/haarcascade_frontalface_default.xml')

#Id = input('enter your id')
Id = '3'
sampleNum = 0
imgs = []
path = "/home/namvh/Documents/do an cuoi cung/file/imageSource/unknown"
valid_images = [".jpg",".jpeg"]
for f in os.listdir(path):
    imgs.append(Image.open(os.path.join(path,f)))
for img in imgs:
    print(img.filename)
    img1 = cv2.imread(img.filename)
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img1, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # incrementing sample number
        sampleNum = sampleNum + 1
        # saving the captured face in the dataset folder

        cv2.imwrite("/home/namvh/Documents/do an cuoi cung/file/images/unknown/unknown." + Id + '.' + str(sampleNum) + ".jpg", img1[y:y + h, x:x + w])
cv2.destroyAllWindows()