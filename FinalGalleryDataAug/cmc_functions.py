from itertools import cycle

from numpy import interp

from CMC import CMC
import numpy as np
import cv2 as cv
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

def cmc_values(rank, num_of_frames):
    i = 0
    score = 0
    values = []
    for i in range(0, len(rank)):
        score = score + float(rank[i]) / float(num_of_frames)
        values.append(score)
    return values


def sLog(data,name):
    file = open(name,"a")
    file.write(str(data))
    file.close()


def plot_cmc(values, title, ranks):
    if ranks > len(values):
        rank = len(values)
    else:
        rank = ranks
    cmc_dict = {
        "at Rank 1": values
    }
    cmc = CMC(cmc_dict)
    cmc.plot(title=title, rank=rank)


def getOriginalClass(label_batch):
    #print(label_batch)
    class_name = 0
    for i, item in enumerate(label_batch):
        if item >0:
            class_name = i
            break
    #print("class_name ="+ str(class_name))
    return class_name


# prende batchMatrix: dim: numero immagini di un batch, numero classi +1
#nella posizione nclassi(quindi 51 esima nel caso di 50 classi) è presente il valore vero della classe associata
def countTry(batchMatrix, num_classes):
    count = np.zeros(shape=num_classes)

    for i in range(len(batchMatrix)):
        trueClass = batchMatrix[i][num_classes]
        for j in range(num_classes):
            if trueClass == batchMatrix[i][j]:
                count[j] += 1
                break
    return count, len(batchMatrix)


# passare tutto il batch_preds e label_batch
def defineImageRank(batch_preds, label_batch):
    batchMatrix = []
    numClasses = int(len(batch_preds[0]))
    rank = np.zeros(shape=(numClasses, 3))

    for i in range(len(batch_preds)):
        # batchMatrix.append([])

        trueClass = getOriginalClass(label_batch[i])

        for j in range(0, numClasses):
            rank[j][0] = j
            rank[j][1] = batch_preds[i][j]
            rank[j][2] = trueClass

        rankO = rank[np.argsort(-rank[:, 1])]

        sortedClass = []
        for j in range(numClasses):
            sortedClass.append(int(rankO[j][0]))

        sortedClass.append(int(rankO[0][2]))
        batchMatrix.insert(i, sortedClass)

    return batchMatrix

# passare tutto il batch_preds e label_batch
def defineImageRank1(batch_preds, label_batch):
    batchMatrix = []
    numClasses = int(len(batch_preds[0]))
    rank = np.zeros(shape=(numClasses, 3))

    for i in range(len(batch_preds)):
        # batchMatrix.append([])

        trueClass = getOriginalClass(label_batch[i])

        for j in range(0, numClasses):
            rank[j][0] = j
            rank[j][1] = batch_preds[i][j]
            rank[j][2] = trueClass

        rankO = rank[np.argsort(rank[:, 1])]

        sortedClass = []
        for j in range(numClasses):
            sortedClass.append(int(rankO[j][0]))

        sortedClass.append(int(rankO[0][2]))
        batchMatrix.insert(i, sortedClass)

    return batchMatrix