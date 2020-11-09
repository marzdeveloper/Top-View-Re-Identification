import csv
import random

path_train = "C:/Users/Daniele/Desktop/TVPR2/train.csv" #path al csv di train
path_test = "C:/Users/Daniele/Desktop/TVPR2/test.csv" #path al csv di test
txt_path = "C:/Users/Daniele/Desktop/TVPR2/txt/prova/"

min = 80 #numero minimo di foto per classe
max = 80 #numero massimo di foto per classe
gallery_photo = 5 #numero di foto per classe nella gallery (minore o uguale di min)

max_id=50
i = 0
count_train = 0
in_file_train = open(path_train)
csvreader_train = csv.reader(in_file_train, delimiter=";")
in_file_test = open(path_test)
csvreader_test = csv.reader(in_file_test, delimiter=";")
gallery_file = open(txt_path+"gallery.txt", "w", newline='')
train_file = open(txt_path+"train.txt", "w", newline='')
val_file = open(txt_path+"val.txt", "w", newline='')
test_file = open(txt_path+"test.txt", "w", newline='')

x = []
dict = {}
people = []

for row in csvreader_train:
    if (len(row) - 3) > (min - 1):
        directory = row[0]
        people.append(int(directory))


pippo = random.sample(people, max_id)

in_file_train.seek(0)

for row in csvreader_train:
        directory = row[0]
        if int(directory) in pippo:
            row = row[3:]
            apache = random.sample(row, min)
            gallery_apache = random.sample(apache, gallery_photo)
            for i, photo in enumerate(gallery_apache):
                gallery_file.write(directory + '/' + photo.strip('_rgb.jpg') + ' ' + str(count_train) + '\n')
            for i, photo in enumerate(apache):
                if i < int(0.7*max):
                    train_file.write(directory + '/' + photo.strip('_rgb.jpg') + ' ' + str(count_train) + '\n')
                else:
                    val_file.write(directory + '/' + photo.strip('_rgb.jpg') + ' ' + str(count_train) + '\n')
            dict[directory] = count_train
            count_train += 1
            if count_train == max_id:
                break

max_test = 50
for row in csvreader_test:
    directory = row[0]
    row = row[3:]
    if directory in dict.keys():
        if len(row) > max_test:
            max_foto = max_test
        else:
            max_foto = len(row)
        apache = random.sample(row, max_foto)
        for photo in apache:
            test_file.write(directory + '/' + photo.strip('_rgb.jpg') + ' ' + str(dict[directory]) + '\n')

print("Numero di classi: " + str(count_train))
