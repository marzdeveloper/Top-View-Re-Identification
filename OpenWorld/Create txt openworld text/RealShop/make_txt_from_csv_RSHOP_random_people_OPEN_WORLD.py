import csv
import random

path = "C:/Users/Daniele/Desktop/RealShop.csv" #path al csv
txt_path = "C:/Users/Daniele/Desktop/Febbraio finale/txt/openword/200id1/"

min = 25 #numero minimo di foto per classe
max = 25 #numero massimo di foto per classe
gallery_photo = 10 #numero di foto per classe nella gallery (minore o uguale di min)
num_intrusi = 20

max_id = 200
i = 0
count = 0
in_file = open(path)
csvreader = csv.reader(in_file, delimiter=";")
train_file = open(txt_path+"train.txt", "w", newline='')
gallery_file = open(txt_path+"gallery.txt", "w", newline='')
val_file = open(txt_path+"val.txt", "w", newline='')
test_file = open(txt_path+"test.txt", "w", newline='')


people = []

for row in csvreader:
    if (len(row) - 3) > (min - 1):
        directory = row[0]
        people.append(int(directory))



if len(people) < max_id + num_intrusi:
    print("Gente richiesta non disponibile, diminuire numero di foto richieste")
    exit(-1)


trainPeople = random.sample(people, max_id)


in_file.seek(0)

for row in csvreader:
        directory = row[0]
        if int(directory) in trainPeople:
            row = row[3:]
            apache = random.sample(row, min)
            gallery_apache = random.sample(apache, gallery_photo)
            for i, photo in enumerate(gallery_apache):
                gallery_file.write(directory + '/' + photo.strip('_rgb.png') + ' ' + str(count) + '\n')
            for i, photo in enumerate(apache):
                if i < int(0.7 * max):
                    train_file.write(directory + '/' + photo.strip('_rgb.png') + ' ' + str(count) + '\n')
                elif i >= int(0.9 * max) and count < max_id - num_intrusi:
                    test_file.write(directory + '/' + photo.strip('_rgb.png') + ' ' + str(count) + '\n')
                elif i >= int(0.7 * max) and i < int(0.9*max):
                    val_file.write(directory + '/' + photo.strip('_rgb.png') + ' ' + str(count) + '\n')
            count += 1
            print(directory)

                
intrusi = random.sample(set(people) - set(trainPeople), num_intrusi)

in_file.seek(0)

count_test = count - num_intrusi
min = int(0.1 * max)
for row in csvreader:
    directory = row[0]
    if int(directory) in intrusi:
        row = row[3:]
        apache = random.sample(row, min)
        for photo in apache:
            test_file.write(directory + '/' + photo.strip('_rgb.png') + ' ' + str(count_test) + '\n')
        count_test += 1
        print(directory)


print("Numero di classi: " + str(count_test) + " e " + str(num_intrusi) + " intrusi")
