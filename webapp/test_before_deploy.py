print("hello")
print("noooooo")

import os

basedir = os.getcwd()

from ml import knn

file_dir = os.path.join(basedir, 'images/')
print("heloooooooo" , file_dir) 

obj = knn.KNNClassifier()
obj.get_random_images()