import csv
import os
import glob

path = "C:/Users/Daniele/Desktop/finale/"
out_path = "C:/Users/Daniele/Desktop/finale.csv"

print("Creando il csv...")
count = 0
out_file = open(out_path, "w", newline='')
csvwriter = csv.writer(out_file, delimiter=";")
for folder in sorted(os.listdir(path), key=int):
    row = []
    row.append(folder)
    row.append(str(count))
    row.append("0")
    for item in os.listdir(path+folder):
        if "_rgb.png" in item:
            row.append(item)
    csvwriter.writerow(row)
    print(folder)
    count += 1
print("Csv creato!")