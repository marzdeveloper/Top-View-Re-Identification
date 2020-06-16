import csv

path = "/home/nico/Scrivania/Computer vision/Progetto/Locale/Dataset_gennaio/csv/result_gennaio.csv" #path al csv
dest_path = "/home/nico/Scrivania/Computer vision/Progetto/Locale/Dataset_gennaio/preprocessed/" #destination folder path
csv_dest_path = "/home/nico/Scrivania/Computer vision/Progetto/Locale/Dataset_gennaio/csv/result_gennaio_100_foto2.csv"

min = 100 #numero minimo di foto per classe
max = 100 #numero massimo di foto per classe

i = 0
count = 0
in_file = open(path)
csvreader = csv.reader(in_file, delimiter=";")
train_file = open(dest_path+"train.txt", "w", newline='')
test_file = open(dest_path+"test.txt", "w", newline='')
out_file = open(csv_dest_path, "w", newline='')
csvwriter = csv.writer(out_file, delimiter=";")

for row in csvreader:
    if (len(row) - 3) > (min - 1):
        count = count + 1
        first_row = []
        first_row.append(row[0])
        new_row = []
        new_row.append(row[0])
        new_row.append(row[1])
        new_row.append(row[2])
        if (len(row) - 3) > max:
            max_foto = max
        else:
            max_foto = (len(row) - 3)
        for i, item in enumerate(row):
            if i in range (3, max_foto + 3):
                new_row.append(item)
                if (i - 2) > int(0.8*max_foto):
                    test_file.write(row[0] + '/' + item.strip('_rgb.png') + ' ' + row[1] + '\n')
                else:
                    train_file.write(row[0] + '/' + item.strip('_rgb.png') + ' ' + row[1] + '\n')
        csvwriter.writerow(new_row)
#out_file_first_row = open(csv_dest_path.strip('.csv') + "_FIRST_ROW.txt", "w", newline='')
#out_file_first_row.write(first_row[0])

print ("Numero di classi: " + str(count))
print ("Manca la prima riga al csv: è " + first_row[0])