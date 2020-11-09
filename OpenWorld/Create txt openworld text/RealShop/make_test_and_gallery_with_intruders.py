import csv
import random
from collections import OrderedDict


path = "C:/Users/Daniele/Desktop/RealShop.csv" #path al csv
txt_path = "C:/Users/Daniele/Desktop/Febbraio finale/txt/openword/200id/"

max = 25        #max num of frames for test id
gallery_photo = 10 #numero di foto per classe nella gallery (minore o uguale di min)

max_id=200
num_intrusi = 20


in_file = open(path)
csvreader = csv.reader(in_file, delimiter=";")
train_file = open(txt_path + "train.txt", "r", newline='')
val_file = open(txt_path+"val.txt", "r", newline='')
test_file = open(txt_path + "testprova1.txt", "w", newline='')
gallery_file = open(txt_path+"gallery_by_train.txt", "w", newline='')

peopleTrain = []
allPeople = []
dictVal = {}
count = 0


for row in csvreader:
    if (len(row) - 3) >= max:
        allPeople.append(int(row[0]))


for row in train_file:
    x = row.split('/')
    peopleTrain.append(x[0])

in_file.seek(0)
train_file.seek(0)
setTrain = OrderedDict.fromkeys(peopleTrain)

val = list(train_file)
passati = []


#creo un dizionario con chiave l'id delle persone in validation e valore i nomi delle foto di validation per poi costruirci la galleria
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


#costruisco la galleria sulle foto della validation
for i, item in enumerate(setTrain.keys()):
    gallery_apache = random.sample(dictVal[item], gallery_photo)
    for photo in gallery_apache:
        gallery_file.write(item + '/' + photo + ' ' + str(i) + '\n')


'''
intrusi = random.sample(set(allPeople) - set(peopleTrain), num_intrusi)


for row in csvreader:
    directory = row[0]
    row = row[3:]
    if directory in setTrain.keys() and count < max_id - num_intrusi:
        if (len(row)) > max:
            max_foto = max
        else:
            max_foto = (len(row))
        apache = random.sample(row, max_foto)
        for photo in apache:
            test_file.write(directory + '/' + photo.strip('_rgb.png') + ' ' + str(count) + '\n')
        count += 1


in_file.seek(0)


for row in csvreader:
    directory = row[0]
    row = row[3:]
    if int(directory) in intrusi:
        if (len(row)) > max:
            max_foto = max
        else:
            max_foto = (len(row))
        apache = random.sample(row, max_foto)
        for photo in apache:
            test_file.write(directory + '/' + photo.strip('_rgb.png') + ' ' + str(count) + '\n')
        count += 1
'''
