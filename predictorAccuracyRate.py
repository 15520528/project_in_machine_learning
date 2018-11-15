import joblib
import os
import sys
import numpy as np

clf = joblib.load('/home/namvh/Documents/do an may hoc/file 1/modelSVM.joblib')
#test = np.load('D://courses_2016-2017//python//MH_vohuynam_15520528//testfeature//Messi//Messi.1.1.npy')

def predict(src):
    #clf = joblib.load('D:\courses_2016-2017\python\MH_vohuynam_15520528\modelSVM.joblib')
    acc = 0
    loss = 0
    for folder in os.listdir(src):
        folder_path = os.path.join(src, folder)
        files = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
        for file in files:
            realLabel = os.path.split(file)[1]
            realLabel = realLabel[:realLabel.find('.')]
            
            test = np.load(file)
            result = clf.predict(np.reshape(test, (1, -1)))
            print(result)
            if(result == realLabel):
                acc = acc+1
            else:
                loss = loss+1
    print("accuracy: ", acc )
    print("loss:",loss )
predict("/home/namvh/Documents/do an may hoc/file 1/testFeature")
#y = clf.predict(np.reshape(test, (1, -1)))
#print(y)
