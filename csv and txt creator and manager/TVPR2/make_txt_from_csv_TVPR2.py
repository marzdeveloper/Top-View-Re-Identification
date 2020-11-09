import csv
import random

path_train = "/home/nico/Scrivania/Computer vision/Progetto/csv/train.csv" #path al csv di train
path_test = "/home/nico/Scrivania/Computer vision/Progetto/csv/test.csv" #path al csv di test
txt_path = "/home/nico/Scrivania/Computer vision/Progetto/csv/"
#csv_dest_path = "C:/Users/Daniele/Desktop/Dataset_gennaio/result_gennaio_1060_foto7.csv"

min = 50 #numero minimo di foto per classe
max = 50 #numero massimo di foto per classe
gallery_photo = 1 #numero di foto per classe nella gallery (minore o uguale di min)

max_id=100
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
#out_file = open(csv_dest_path, "w", newline='')
#csvwriter = csv.writer(out_file, delimiter=";")
x = []
dict = {}

for row in csvreader_train:
    if (len(row) - 3) > (min - 1):
        if (len(row) - 3) > max:
            max_foto = max
        else:
            max_foto = (len(row) - 3)
        directory = row[0]
        row = row[3:]
        apache = random.sample(row, min)
        gallery_apache = random.sample(apache, gallery_photo)
        for i, photo in enumerate(gallery_apache):
            gallery_file.write(directory + '/' + photo.strip('_rgb.jpg') + ' ' + str(count_train) + '\n')
        for i, photo in enumerate(apache):
            if i < int(0.8*max):#max o max_foto ?
                train_file.write(directory + '/' + photo.strip('_rgb.jpg') + ' ' + str(count_train) + '\n')
            else:
                val_file.write(directory + '/' + photo.strip('_rgb.jpg') + ' ' + str(count_train) + '\n')
        dict[directory] = count_train
        count_train += 1
        if count_train == max_id:
            break

for row in csvreader_test:
    directory = row[0]
    row = row[3:]
    if directory in dict.keys():
        if (len(row)) > max:
            max_foto = max
        else:
            max_foto = (len(row))
        apache = random.sample(row, max_foto)
        for photo in apache:
            test_file.write(directory + '/' + photo.strip('_rgb.jpg') + ' ' + str(dict[directory]) + '\n')
#out_file_first_row = open(csv_dest_path.strip('.csv') + "_FIRST_ROW.txt", "w", newline='')
#out_file_first_row.write(first_row[0])

print("Numero di classi: " + str(count_train))
#print("Manca la prima riga al csv: è " + first_row[0])
