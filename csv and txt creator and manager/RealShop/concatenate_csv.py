import csv

dir_path = "/home/nico/Scrivania/Computer vision/Progetto/Dataset_3/Dataset/gennaio/csv/" #folder che contiene i csv da unire
dest_csv_path = "/home/nico/Scrivania/Computer vision/Progetto/Locale/Dataset_gennaio/csv/result_gennaio.csv" #csv di destinazione
#list of csv to concatenate
file = ["2019-12-30.csv", "2019-12-31.csv", "2020-01-02.csv", "2020-01-03.csv", "2020-01-04.csv", "2020-01-05.csv", "2020-01-06.csv", "2020-01-07.csv", "2020-01-08.csv", "2020-01-09.csv", "2020-01-10.csv", "2020-01-11.csv", "2020-01-12.csv", "2020-01-13.csv", "2020-01-14.csv", "2020-01-15.csv", "2020-01-16.csv", "2020-01-17.csv", "2020-01-18.csv", "2020-01-19.csv", "2020-01-20.csv", "2020-01-21.csv", "2020-01-22.csv", "2020-01-23.csv", "2020-01-24.csv", "2020-01-25.csv", "2020-01-26.csv", "2020-01-27.csv", "2020-01-28.csv", "2020-01-29.csv", "2020-01-30.csv", "2020-01-31.csv"]

i = 0
out_file = open(dest_csv_path, "w", newline='')
csvwriter = csv.writer(out_file, delimiter=";")
first_row_file = open(dir_path + "result_" + file[len(file) - 1])
firstrowreader = csv.reader(first_row_file, delimiter=";")
for row in firstrowreader:
    if len(row) == 1:
        csvwriter.writerow(row)
for i in range(0, len(file)):
    in_file = open(dir_path + "result_" + file[i])
    csvreader = csv.reader(in_file, delimiter=";")
    for row in csvreader:
        if len(row) > 1:
            csvwriter.writerow(row)

print ("Csv creato!")