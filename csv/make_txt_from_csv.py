import csv
import os

i = 0
path = "C:/Users/Daniele/Desktop/Nuova cartella (4)/2020-01-03_clean/result_2020-01-03.csv"

#for file in os.listdir(path):
in_file = open(path)
csvreader = csv.reader(in_file, delimiter=";")
train_file = open(path+"train.txt", "w", newline='')
test_file = open(path+"test.txt", "w", newline='')

for row in csvreader:
    if len(row) > 1:
        for i, item in enumerate(row):
            if i > 2:
                if i > int(0.8*(len(row) - 3)):
                    test_file.write(row[0] + '/' + item.strip('_rgb.png') + ' ' + row[1] + '\n')
                else:
                    train_file.write(row[0] + '/' + item.strip('_rgb.png') + ' ' + row[1] + '\n')
