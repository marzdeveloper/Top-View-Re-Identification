import csv
import random

path = "C:/Users/Daniele/Desktop/finale.csv" #path al csv
txt_path = "C:/Users/Daniele/Desktop/Febbraio finale/txt/finale/50id_50foto/"
#csv_dest_path = "C:/Users/Daniele/Desktop/Dataset_gennaio/result_gennaio_1060_foto7.csv"

min = 50 #numero minimo di foto per classe
max = 50 #numero massimo di foto per classe

max_id=50
i = 0
count = 0
in_file = open(path)
csvreader = csv.reader(in_file, delimiter=";")
train_file = open(txt_path+"train.txt", "w", newline='')
val_file = open(txt_path+"val.txt", "w", newline='')
test_file = open(txt_path+"test.txt", "w", newline='')
#out_file = open(csv_dest_path, "w", newline='')
#csvwriter = csv.writer(out_file, delimiter=";")

for row in csvreader:
    if (len(row) - 3) > (min - 1):
        if (len(row) - 3) > max:
            max_foto = max
        else:
            max_foto = (len(row) - 3)
        directory = row[0]
        row = row[3:]
        apache = random.sample(row, min)
        for i, photo in enumerate(apache):
            if i < int(0.7*max):#max o max_foto ?
                train_file.write(directory + '/' + photo.strip('_rgb.png') + ' ' + str(count) + '\n')
            elif i >= int(0.9*max):
                test_file.write(directory + '/' + photo.strip('_rgb.png') + ' ' + str(count) + '\n')
            else:
                val_file.write(directory + '/' + photo.strip('_rgb.png') + ' ' + str(count) + '\n')
        count += 1
        print(directory)
        if count == max_id:
            break
#out_file_first_row = open(csv_dest_path.strip('.csv') + "_FIRST_ROW.txt", "w", newline='')
#out_file_first_row.write(first_row[0])

print("Numero di classi: " + str(count))
#print("Manca la prima riga al csv: è " + first_row[0])