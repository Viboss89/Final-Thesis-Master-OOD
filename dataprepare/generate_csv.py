import csv
import os
import glob
import cv2
import matplotlib.pyplot as plt
import random

"""
Script to generate csv file with labels of the images
"""

# TODO cambiar labels str to integers

path_data = "../ZhangLabData"
path_csv_file = "../ZhangLabData/labels.csv"

frac = [0.7, 0.2, 0.1]


def unique(list1):
 
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = (list(list_set))

    return unique_list


# lists of paths and classes
paths_list = []
classes_list = []

for root, dirs, files in os.walk(path_data, topdown=False):

    for dir in dirs:
        paths = glob.glob(path_data+"/"+dir+"/*.jpeg")
        
        for path in paths:
            paths_list.append(path)
            classes_list.append(dir)
print(classes_list)
# separate into datasets
labels = unique(classes_list)   
list_shuffle = list(zip(paths_list, classes_list))
random.shuffle(list_shuffle)

# make list separate
paths_list, classes_list = zip(*list_shuffle)

f = open(path_data+"/train.csv", "w")
train_path = paths_list[:int(len(paths_list)*frac[0])]
train_labels = classes_list[:int(len(paths_list)*frac[0])]
for path, label in zip(train_path,train_labels):
    f.write(path+","+str(labels.index(label))+"\n")
f.close()

f = open(path_data+"/test.csv", "w")
test_path = paths_list[int(len(paths_list)*frac[0]):int(len(paths_list)*frac[0])+int(len(paths_list)*frac[1])]
test_labels = classes_list[int(len(paths_list)*frac[0]):int(len(paths_list)*frac[0])+int(len(paths_list)*frac[1])]
for path, label in zip(test_path,test_labels):
    f.write(path+","+str(labels.index(label))+"\n")
f.close()

f = open(path_data+"/val.csv", "w")
val_path = paths_list[int(len(paths_list)*frac[0])+int(len(paths_list)*frac[1]):]
val_labels = classes_list[int(len(paths_list)*frac[0])+int(len(paths_list)*frac[1]):]
for path, label in zip(val_path,val_labels):
    f.write(path+","+str(labels.index(label))+"\n")
f.close()
