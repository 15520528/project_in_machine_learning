
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
import numpy as np
import sys
import os
from PIL import Image

Image.LOAD_TRUNCATED_IMAGES=True
print("[+] Setup model")
base_model = VGG16(weights='imagenet', include_top=True)
out = base_model.get_layer("fc2").output
model = Model(inputs=base_model.input, outputs=out)
import cv2

path = "/home/namvh/Documents/do an may hoc/file/pridict/image"
imgs = []

def save_feature(save_path, feature):    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("[+]Save extracted feature to file : ", save_path)
    np.save(save_path, feature)


for f in os.listdir(path):
    imgs.append(Image.open(os.path.join(path,f)))
for img in imgs:
    img_path = img.filename
    print(img_path)
    save_path = img_path.replace("image", "feature").replace(".jpg", ".npy")            
                    
    imga = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(imga)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    
    feature = model.predict(img_data)    
    #print("[+] Extract feature from image : ", img_path)
    save_feature(save_path, feature)  
cv2.destroyAllWindows()