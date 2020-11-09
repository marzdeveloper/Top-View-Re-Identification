import csv
import random
from collections import OrderedDict


path_test = "C:/Users/Daniele/Desktop/TVPR2/test.csv" #path al csv di test
txt_path = "C:/Users/Daniele/Desktop/TVPR2/txt/openworld/200id/"

max = 50        #max num of frames for test id
gallery_photo = 5 #numero di foto per classe nella gallery (minore o uguale di min)

max_id=200
num_intrusi = 60


in_file_test = open(path_test)
csvreader_test = csv.reader(in_file_test, delimiter=";")
train_file = open(txt_path + "train.txt", "r", newline='')
test_file = open(txt_path + "test60.txt", "w", newline='')


peopleTrain = []
allPeople = []
count_test = 0


for row in csvreader_test:
    if (len(row) - 3) >= max:
        allPeople.append(int(row[0]))


for row in train_file:
    x = row.split('/')
    peopleTrain.append(x[0])

in_file_test.seek(0)

setTrain = OrderedDict.fromkeys(peopleTrain)

for row in csvreader_test:
    directory = row[0]
    row = row[3:]
    if directory in setTrain.keys() and count_test < max_id - num_intrusi:
        if (len(row)) > max:
            max_foto = max
        else:
            max_foto = (len(row))
        apache = random.sample(row, max_foto)
        for photo in apache:
            test_file.write(directory + '/' + photo.strip('_rgb.jpg') + ' ' + str(count_test) + '\n')
        count_test += 1


in_file_test.seek(0)

intrusi = random.sample(set(allPeople) - set(peopleTrain), num_intrusi)


for row in csvreader_test:
    directory = row[0]
    row = row[3:]
    if int(directory) in intrusi:
        if (len(row)) > max:
            max_foto = max
        else:
            max_foto = (len(row))
        apache = random.sample(row, max_foto)
        for photo in apache:
            test_file.write(directory + '/' + photo.strip('_rgb.jpg') + ' ' + str(count_test) + '\n')
        count_test += 1

