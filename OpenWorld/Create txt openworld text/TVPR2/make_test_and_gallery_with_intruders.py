import csv
import random
from collections import OrderedDict


path_train = "C:/Users/Daniele/Desktop/TVPR2/train.csv" #path al csv di train
path_test = "C:/Users/Daniele/Desktop/TVPR2/test.csv" #path al csv di test
txt_path = "C:/Users/Daniele/Desktop/TVPR2/txt/openworld/100id_last/"
#csv_dest_path = "C:/Users/Daniele/Desktop/Dataset_gennaio/result_gennaio_1060_foto7.csv"

max = 50        #max num of frames for test id
gallery_photo = 15 #numero di foto per classe nella gallery (minore o uguale di min)

max_id=100
num_intrusi = 30


in_file_test = open(path_test)
csvreader_test = csv.reader(in_file_test, delimiter=";")
train_file = open(txt_path + "train.txt", "r", newline='')
val_file = open(txt_path+"val.txt", "r", newline='')
test_file = open(txt_path + "/1/test.txt", "w", newline='')
gallery_file = open(txt_path+"/1/gallery_by_val.txt", "w", newline='')

peopleTrain = []
allPeople = []
dictVal = {}
count_test = 0


for row in csvreader_test:
    if (len(row) - 3) >= max:
        allPeople.append(int(row[0]))


for row in train_file:
    x = row.split('/')
    peopleTrain.append(x[0])

in_file_test.seek(0)

setTrain = OrderedDict.fromkeys(peopleTrain)

val = list(val_file)
passati = []

for i in range(len(val)):
    row = val[i]
    x = row.split('/')
    key = str(x[0])
    lista = []
    if key not in passati:
        while val[i].split('/')[0] == key:
            lista.extend(val[i].replace(' ', '/').split('/')[1:2])
            if i +1 < len(val):
                i += 1
            else:
                break
        dictVal[key] = lista
        passati.append(key)

for i, item in enumerate(setTrain.keys()):
    gallery_apache = random.sample(dictVal[item], gallery_photo)
    for photo in gallery_apache:
        gallery_file.write(item + '/' + photo + ' ' + str(i) + '\n')

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

