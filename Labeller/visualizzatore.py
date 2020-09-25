import cv2
import csv

csv_path = "/home/nico/Scrivania/Computer vision/Progetto/Locale/Dataset_gennaio/csv/result_gennaio.csv"
folder_path = "/home/nico/Scrivania/Computer vision/Progetto/Locale/Dataset_gennaio/preprocessed/"

csv_file = open(csv_path)
csvreader = csv.reader(csv_file, delimiter=";")
for row in csvreader:
    if len(row) > 1:
        image_path = folder_path + row[0] + "/" + row[3]
        img = cv2.imread(image_path)
        cv2.imshow(row[1], img)
        cv2.waitKey(0)