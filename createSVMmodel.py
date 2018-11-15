import numpy as np
import sys
import os
from sklearn.svm import LinearSVC
import joblib

def LoadFeature(src):
    print("[+] Loading data...")
    train_set = []
    train_lable = []
    list_folder = os.listdir(src)
    for lable, folder in enumerate(list_folder):
        list_file = os.listdir(os.path.join(src, folder))
        for file in list_file:
            a = os.path.join(src, folder, file)
            feature = np.load(a)
            train_set.append(feature[0])
            train_lable.append(folder)
    print("[+] Load data finished")
    return train_set, train_lable

def Clustering(trainSet, trainLable):
    print("[!] Clustering data...")
    clf = LinearSVC()
    clf.fit(trainSet, trainLable)
    print("Training finished")
    return clf

def SaveModel(model, name):
    file_name = name + ".joblib"
    print("[+] Saving model to file: ", file_name)
    joblib.dump(model, file_name)
    
            
if __name__=="__main__":
    src = "/home/namvh/Documents/do_an_cuoi_cung/file/trainFeature"
    train_set, train_lable = LoadFeature(src)
    clf = Clustering(train_set, train_lable)
    SaveModel(clf, "/home/namvh/Documents/do_an_cuoi_cung/file/modelSVM")