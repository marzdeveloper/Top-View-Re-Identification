import csv
import os

i = 0
path = "/home/nico/Scrivania/Computer vision/Progetto/Locale/121_id/csv/result_121_id.csv" #path al csv
dest_path = "/home/nico/Scrivania/Computer vision/Progetto/Locale/121_id/preprocessed/" #destination folder path
min = 20 #numero minimo di foto per classe
max = 40 #numero massimo di foto per classe

#for file in os.listdir(path):
in_file = open(path)
csvreader = csv.reader(in_file, delimiter=";")
train_file = open(dest_path+"train.txt", "w", newline='')
test_file = open(dest_path+"test.txt", "w", newline='')

for row in csvreader:
    if (len(row) - 3) > (min -1):
        if (len(row) - 3) > max:
            max_foto = max
        else:
            max_foto = (len(row) - 3)
        for i, item in enumerate(row):
            if i in range (3, max_foto + 3):
                if (i - 2) > int(0.8*max_foto):
                    test_file.write(row[0] + '/' + item.strip('_rgb.png') + ' ' + row[1] + '\n')
                else:
                    train_file.write(row[0] + '/' + item.strip('_rgb.png') + ' ' + row[1] + '\n')
