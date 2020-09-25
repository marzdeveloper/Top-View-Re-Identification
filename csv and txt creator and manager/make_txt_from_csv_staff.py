import csv
import random

path = "D:/DATASET/csv_con_staff_separato/csv genn staff separato/result_.csv" #path al csv
txt_path = "D:/DATASET/csv_con_staff_separato/csv genn staff separato/"
csv_dest_path = "D:/DATASET/csv_con_staff_separato/csv genn staff separato/result_gennaio_staff_separato.csv"

min = 100 #numero minimo di foto per classe
max = 100 #numero massimo di foto per classe

max_id_train=46 #max num of train id
max_id_test=100 #max num of test id

i = 0
count = 0   #contatore id aggiunti

in_file = open(path)
csvreader = csv.reader(in_file, delimiter=";")
train_file = open(txt_path+"train.txt", "w", newline='')
val_file = open(txt_path+"val.txt", "w", newline='')
test_file = open(txt_path+"test.txt", "w", newline='')

staff = [1,8,9,144] #contiene gli id dello staff che si vuole aggiungere
staff_train = []    #id originale dello staff
people_train = []   #id originale delle persone
staff_dict = {}     #associa staff a staff train


#scan per persone normali
for row in csvreader:
    if (len(row) - 3) > (min - 1):
        if (len(row) - 3) > max:
            max_foto = max
        else:
            max_foto = (len(row) - 3)
        directory = row[1]
        original = row[0]
        row = row[3:]
        apache = random.sample(row, min)
        if int(directory) not in staff:
            for i, photo in enumerate(apache):
                people_train.append(int(original))
                if i < int(0.7*max):#max o max_foto ?
                    train_file.write(original + '/' + photo.strip('_rgb.png') + ' ' + str(count) + '\n')
                elif i >= int(0.9*max):
                    test_file.write(original + '/' + photo.strip('_rgb.png') + ' ' + str(count) + '\n')
                else:
                    val_file.write(original + '/' + photo.strip('_rgb.png') + ' ' + str(count) + '\n')
            count += 1
        if count == max_id_train:
            break

max_id_train = 50
in_file.seek(0)
#scan per train e val dello staff
for row in csvreader:
    if (len(row) - 3) > (min - 1):
        if (len(row) - 3) > max:
            max_foto = max
        else:
            max_foto = (len(row) - 3)
        directory = row[1]
        original = row[0]
        if int(directory) in staff:
            row = row[3:]
            apache = random.sample(row, min)
            for i, photo in enumerate(apache):
                staff_train.append(int(original))
                if i < int(0.8*max):#max o max_foto ?
                    train_file.write(original + '/' + photo.strip('_rgb.png') + ' ' + str(count) + '\n')
                else:
                    val_file.write(original + '/' + photo.strip('_rgb.png') + ' ' + str(count) + '\n')
            staff_dict[directory] = count
            staff.remove(int(directory))
            count += 1
        if count == max_id_train:
            break


#numero foto test per id
min = 10 #numero minimo di foto per classe
max = 10 #numero massimo di foto per classe
staff = [1,8,9,144]

#scan per aggiungere persone nel test che non compaiono nel train e val
in_file.seek(0)
for row in csvreader:
    if (len(row) - 3) > (min - 1):
        if (len(row) - 3) > max:
            max_foto = max
        else:
            max_foto = (len(row) - 3)
        directory = row[1]
        original = row[0]
        if not (int(original) in people_train and int(directory) in staff):
            row = row[3:]
            apache = random.sample(row, min)
            for photo in apache:
                test_file.write(original + '/' + photo.strip('_rgb.png') + ' ' + str(count) + '\n')                #mette nel test la gente normale
            count += 1
        if count == max_id_test:
            break

min = 50 #numero minimo di foto per classe
max = 50 #numero massimo di foto per classe
in_file.seek(0)
#scan per aggiungere giorni dello staff nel test che non compaiono nel train e val
for row in csvreader:
    if (len(row) - 3) > (min - 1):
        if (len(row) - 3) > max:
            max_foto = max
        else:
            max_foto = (len(row) - 3)
        directory = row[1]
        original = row[0]
        if directory in staff:
            if not (int(original) in staff_train):
                row = row[3:]
                apache = random.sample(row, min)
                for photo in apache:
                    test_file.write(original + '/' + photo.strip('_rgb.png') + ' ' + str(staff_dict[directory]) + '\n')            #mette nel test lo staff in giorni diversi dal training
                staff.remove(int(directory))


print("Numero di classi: " + str(count))
print("Staff dict", staff_dict)