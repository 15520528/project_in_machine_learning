from keras.engine import  Model
from keras.layers import Input
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.preprocessing import image
import numpy as np
import sys                                                                                                                                                                                                                                                  
import os
from PIL import ImageFile

'''
ImageFile.LOAD_                                                                                                                                                         TRUNCATED_IMAGES=True
print("[+] Setup model")
layer_name = 'fc1' # edit this line
vgg_model = VGGFace()
out = vgg_model.get_layer(layer_name).output
model = Model(vgg_model.input, out)                                                                                                                                                                                                                                                                                                                                                         
'''

# tensorflow
model = VGGFace() # default : VGG16 , you can use model='resnet50' or 'senet50'


def save_feature(save_path, feature):    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("[+]Save extracted feature to file : ", save_path)
    np.save(save_path, feature)

def extract_features(src):
    with open(src, "r") as file:
        for i,line in enumerate(file):
            img_path = line[:-1]
            print("[+] Read image  : ", img_path," id : ", i)
            if os.path.isfile(img_path) and img_path.find(".jpg") != -1:            
                save_path = img_path.replace("images", "testFeature").replace(".jpg", ".npy")            
                      
                img = image.load_img(img_path, target_size=(224, 224))
                img_data = image.img_to_array(img)
                img_data = np.expand_dims(img_data, axis=0)
                img_data = utils.preprocess_input(img_data, version=1) # or version=2

                print("[+] Extract feature from image : ", img_path)
                feature = model.predict(img_data)

                save_feature(save_path, feature)
            

if __name__=="__main__":
    src = "/home/namvh/Documents/do an may hoc/file 1/dataSet/test.txt"
    print(src)
    extract_features(src)


