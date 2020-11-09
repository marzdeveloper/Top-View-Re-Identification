import csv
import random
from collections import OrderedDict

path_test = "C:/Users/Daniele/Desktop/TVPR2/test.csv" #path al csv di test
txt_path = "C:/Users/Daniele/Desktop/TVPR2/txt/openworld/100id/"
#csv_dest_path = "C:/Users/Daniele/Desktop/Dataset_gennaio/result_gennaio_1060_foto7.csv"

max = 50 #numero massimo di foto per classe di test


in_file_test = open(path_test)
csvreader_test = csv.reader(in_file_test, delimiter=";")
train_file = open(txt_path+"train.txt", "r", newline='')
test_file = open(txt_path+"testx.txt", "w", newline='')

count_test = 0
allPeople = []

for row in train_file:
    x = row.split('/')
    allPeople.append(x[0])


setTrain = OrderedDict.fromkeys(allPeople)

for row in csvreader_test:
    directory = row[0]
    row = row[3:]
    if directory in setTrain.keys():
        if (len(row)) > max:
            max_foto = max
        else:
            max_foto = (len(row))
        apache = random.sample(row, max_foto)
        for photo in apache:
            test_file.write(directory + '/' + photo.strip('_rgb.jpg') + ' ' + str(count_test) + '\n')
        count_test += 1



