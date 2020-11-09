import csv
import random
from collections import OrderedDict


path = "C:/Users/Daniele/Desktop/RealShop.csv" #path al csv
txt_path = "C:/Users/Daniele/Desktop/Febbraio finale/txt/openword/200id1/"

max = 3        #max num of frames for test

max_id=200
num_intrusi = 100


in_file = open(path)
csvreader = csv.reader(in_file, delimiter=";")
#train_file = open(txt_path + "train.txt", "r", newline='')
test_file_read = open(txt_path + "/test.txt", "r", newline='')
test_file = open(txt_path + "/100intrusi/test.txt", "w", newline='')


peopleTrain = []
allPeople = []
count = 0


for row in csvreader:
    if (len(row) - 3) >= max:
        allPeople.append(int(row[0]))

'''
for row in train_file:
    x = row.split('/')
    peopleTrain.append(x[0])
'''

print("Elementi massimi:", len(allPeople))

if len(allPeople) < max_id + num_intrusi:
    print("Elementi insufficiente, diminuire intrusi o numero di frames")
    exit(-1)

in_file.seek(0)

setTrain = OrderedDict.fromkeys(peopleTrain)


for row in test_file_read:
    directory = row.strip("/")
    if count < 3 * (max_id - num_intrusi):
        peopleTrain.append(directory)
        test_file.write(row)
    count += 1



intrusi = random.sample(set(allPeople) - set(peopleTrain), num_intrusi)

count = max_id - num_intrusi

for row in csvreader:
    directory = row[0]
    row = row[3:]
    if int(directory) in intrusi:
        apache = random.sample(row, max)
        for photo in apache:
            test_file.write(directory + '/' + photo.strip('_rgb.png') + ' ' + str(count) + '\n')
        count += 1

