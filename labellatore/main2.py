import tkinter as tk
import os
from _datetime import datetime
from MainWindow import MainWindow

def main():
    #Pil da problemi con alcune versioni di python, con la 3.4 va bene
    # il percorso ai risultati va specificato se si vuole continuare il lavoro va specificata, altrimenti no
    #se metti una cartella gia esistente ma vuota da errore, la cartella se esiste DEVE essere piena
    # per comodità imporre:
    # nome_file_dei_risultati = nome_cartella_giornata

    #TASTO DX ESCLUDE TUTTA LA RIGA
    #TASTO SX 1 CLICK ESCLUDE SINGOLA FOTO
    #TASTO SX 2 CLICK MODIFICA LA LUMINOSITA
    #PREMERE LA ROTELLA APRE L IMMAGINE -per poterla vedere meglio-
    #ABILITARE LA SPUNTA DELLA CASELLA METTE A 1 IL VALORE DELLA 3 COLONNA DEL CSV (potrebbe servire per segnare il progresso)

    #USARE LE SPUNTE QUANDO SONO PRESENTI PIU PERSONE PER RIGA

    pathToImages = "C:/Users/Daniele/Downloads/UnivPm/Magistrale/1 anno/Computer Vision & Deep Learning/Progetto/dataset/gennaio/2020-01-21_clean"
    pathToResults = "C:/Users/Daniele/Desktop/label/21-01.csv"
    numOfFolder = 200
    min_images = 0
    start_time = 0

    image_path = pathToImages
    result = pathToResults
    n = numOfFolder


    last_id = -1
    if os.path.exists(result):
        with open(result, 'r') as f:
            lines = f.read().splitlines()
            last_line = lines[0]
            last_id = int(last_line)



    print("BOOOOOOOOOO="+os.path.dirname(__file__))
    images_folders = [k for k in sorted(os.listdir(os.path.join(os.path.dirname(__file__), image_path)), key=int)
                      if len(os.listdir(os.path.join(os.path.dirname(__file__), image_path, k)))/2 >= min_images and
                      int(sorted(os.listdir(os.path.join(os.path.dirname(__file__), image_path, k)),
                                 key=lambda s: datetime.strptime(s[:12], '%H-%M-%S-%f'))[0].split('-')[0]) >= start_time
                      and int(k) > last_id]

    images_folders = images_folders[:min(n, len(images_folders))]

    if not images_folders:
        print("no image folder")
    else:
        root = tk.Tk()
        root.title("pyMultipleImgAnnot")
        MainWindow(root, result, image_path, images_folders, n)
        root.mainloop()




if __name__ == "__main__":
    main()