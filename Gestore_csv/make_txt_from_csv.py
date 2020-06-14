import csv

i = 0
count = 0

path = "/home/nico/Scrivania/Computer vision/Progetto/Locale/Dataset_gennaio/csv/result_gennaio.csv" #path al csv
dest_path = "/home/nico/Scrivania/Computer vision/Progetto/Locale/Dataset_gennaio/preprocessed/" #destination folder path

min = 100 #numero minimo di foto per classe
max = 100 #numero massimo di foto per classe

#for file in os.listdir(path):
in_file = open(path)
csvreader = csv.reader(in_file, delimiter=";")
train_file = open(dest_path+"train.txt", "w", newline='')
test_file = open(dest_path+"test.txt", "w", newline='')

for row in csvreader:
    if (len(row) - 3) > (min - 1):
        count = count + 1
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

print ("Numero di classi: " + str(count))