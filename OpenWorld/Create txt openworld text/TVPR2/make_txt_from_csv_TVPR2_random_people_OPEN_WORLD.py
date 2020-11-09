import csv
import random

path_train = "C:/Users/Daniele/Desktop/TVPR2/train.csv" #path al csv di train
path_test = "C:/Users/Daniele/Desktop/TVPR2/test.csv" #path al csv di test
txt_path = "C:/Users/Daniele/Desktop/TVPR2/txt/openworld/100idora/"

min = 50 #numero minimo di foto per classe
max = 50 #numero massimo di foto per classe
gallery_photo = 15 #numero di foto per classe nella gallery (minore o uguale di min)

max_id=100
num_intrusi = 30
count_train = 0

in_file_train = open(path_train)
csvreader_train = csv.reader(in_file_train, delimiter=";")
in_file_test = open(path_test)
csvreader_test = csv.reader(in_file_test, delimiter=";")
gallery_file = open(txt_path+"gallery.txt", "w", newline='')
train_file = open(txt_path+"train.txt", "w", newline='')
val_file = open(txt_path+"val.txt", "w", newline='')
test_file = open(txt_path+"test.txt", "w", newline='')


dict = {}
people = []   #lista che contiene tutti gli id delle persone che hanno almeno min frames
allPeople = []  #lista che contiene tutti gli id delle persone

for row in csvreader_train:
    allPeople.append(int(row[0]))
    if (len(row) - 3) > (min - 1):
        directory = row[0]
        people.append(int(directory))

print("Elementi massimi:", len(people))

if len(people) < max_id:
    print("Elementi insufficiente, diminuire intrusi o numero di frames")
    exit(-1)


# scelgo randomicamente max_id persone dalla lista people
peopleTrain = random.sample(people, max_id)

in_file_train.seek(0)

for row in csvreader_train:
    if (len(row) - 3) > (min - 1):
        directory = row[0]
        if int(directory) in peopleTrain:
            row = row[3:]
            apache = random.sample(row, min)
            gallery_apache = random.sample(apache, gallery_photo)
            for photo in gallery_apache:
                gallery_file.write(directory + '/' + photo.strip('_rgb.jpg') + ' ' + str(count_train) + '\n')
            for i, photo in enumerate(apache):
                if i < int(0.7 * max):
                    train_file.write(directory + '/' + photo.strip('_rgb.jpg') + ' ' + str(count_train) + '\n')
                else:
                    val_file.write(directory + '/' + photo.strip('_rgb.jpg') + ' ' + str(count_train) + '\n')
            dict[directory] = count_train
            count_train += 1
            if count_train == max_id:
                break

count_test = 0
max = 50        #max num of frames for test id

for row in csvreader_test:
    directory = row[0]
    row = row[3:]
    if directory in dict.keys() and count_test < count_train - num_intrusi:
        if (len(row)) > max:
            max_foto = max
        else:
            max_foto = (len(row))
        apache = random.sample(row, max_foto)
        for photo in apache:
            test_file.write(directory + '/' + photo.strip('_rgb.jpg') + ' ' + str(dict[directory]) + '\n')
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


print("Numero di classi: " + str(count_train))
