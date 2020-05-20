import csv
import os

i = 0
path = "C:/Users/Daniele/Desktop/Nuova cartella (4)/csv_gennaio/"
path1 = "C:/Users/Daniele/Desktop/Nuova cartella (4)/csv_genn/"

for file in os.listdir(path):
    in_file = open(path + file)
    out_file = open(path1 + file, "w", newline='')
    csvreader = csv.reader(in_file, delimiter=";")
    csvwriter = csv.writer(out_file, delimiter=";")
    for row in csvreader:
        if len(row) > 1:
            row[1] = i
            csvwriter.writerow(row)
            i += 1
        else:
            csvwriter.writerow(row)
