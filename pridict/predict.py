
import tkinter as tk
from tkinter import filedialog
import cv2
import matplotlib.pyplot as plt
from keras.engine import  Model
from keras.layers import Input
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.preprocessing import image
from PIL import ImageFile
from PIL import Image
import joblib
import numpy as np
import face_recognition
import os
import glob
cascadePath = "/home/namvh/Documents/do an may hoc/file 1/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
clf = joblib.load('/home/namvh/Documents/do_an_cuoi_cung/file/pridict/modelSVM.joblib')


# tensorflow
model = VGGFace() # default : VGG16 , you can use model='resnet50' or 'senet50'


font = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255, 0, 0)

def save_feature(save_path, feature):    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("[+]Save extracted feature to file : ", save_path)
    np.save(save_path, feature)

root = tk.Tk()
def buttonClick():
    """ handle button click event and output text from entry area"""
    print('hello')    # do here whatever you want
    root.filename = filedialog.askopenfilename(initialdir="/home/namvh/Documents/do_an_cuoi_cung/file/imageTest", title="Select file",
                                               filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    im=cv2.imread(root.filename)
    
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    
    #luu anh
    files = glob.glob('/home/namvh/Documents/do_an_cuoi_cung/file/pridict/image/*')
    for f in files:
        os.remove(f)
    files = glob.glob('/home/namvh/Documents/do_an_cuoi_cung/file/pridict/feature/*')
    for f in files:
        os.remove(f)
    sampleNum = 0   
    for(x,y,w,h) in faces:
        sampleNum = sampleNum + 1
        cv2.imwrite("/home/namvh/Documents/do_an_cuoi_cung/file/pridict/image/img." + '1.' + str(sampleNum) + ".jpg", im[y:y + h, x:x + w])

    #extra feature
    path = "/home/namvh/Documents/do_an_cuoi_cung/file/pridict/image"
    imgs = []
    for f in os.listdir(path):
        imgs.append(Image.open(os.path.join(path,f)))
    for img in imgs:
        img_path = img.filename
        print(img_path)
        save_path = img_path.replace("image", "feature").replace(".jpg", ".npy")            
                        
        imga = image.load_img(img_path, target_size=(224, 224))
        img_data = image.img_to_array(imga)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = utils.preprocess_input(img_data, version=1) # or version=2

        
        feature = model.predict(img_data)    
        #print("[+] Extract feature from image : ", img_path)
        save_feature(save_path, feature)
    #load feature
    path = "/home/namvh/Documents/do_an_cuoi_cung/file/pridict/feature"
    imgs = []
    results = {}
    #xac dinh danh tính
    for f in os.listdir(path):
        filePath = os.path.join(path,f)
        test = np.load(filePath)
        result = clf.predict(np.reshape(test, (1, -1)))
        #chi so i
        i = f[6:]
        i = i[:i.find('.')]
        print(filePath, i , result)

        results[int(i)] = result[0]
    print(results)
    i=1
    #hiển thị danh tính
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
        cv2.putText(im, results[i], (x, y + h), font, 1, fontcolor)
        i = i+1
    plt.imshow(im)
    plt.show()
frame = tk.Frame(root, width=400, height=400, background="bisque")
frame.place(relx=.5, rely=.5, anchor="c")

button = tk.Button(frame,
                text="Choose a file",
                fg="red",
                command=buttonClick)
button.pack(side=tk.LEFT)
root.mainloop()
